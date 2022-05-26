import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.custom_callbacks import (
    DetectOverfittingCallback,
    TimeHistoryCallback,
)
from astronet.metrics import (
    DistributedWeightedLogLoss,
    WeightedLogLoss,
)
from astronet.t2.funcmodel import build_model
from astronet.t2.model import T2Model
from astronet.utils import (
    astronet_logger,
    find_optimal_batch_size,
    load_dataset,
)

try:
    log = astronet_logger(__file__)
    log.info("=" * shutil.get_terminal_size((80, 20))[0])
    log.info(f"File Path: {Path(__file__).absolute()}")
    log.info(f"Parent of Directory Path: {Path().absolute().parent}")
except Exception as e:
    print(f"{e}: Seems you are running from a notebook...")
    __file__ = f"{Path().resolve().parent}/astronet/t2/train.py"
    log = astronet_logger(__file__)

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"


class Training(object):
    def __init__(
        self, epochs, dataset, model, redshift, augmented, avocado, testset, fink
    ):
        self.epochs = epochs
        self.dataset = dataset
        self.model = model
        self.redshift = redshift
        self.augmented = augmented
        self.avocado = avocado
        self.testset = testset
        self.fink = fink

    def __call__(self):
        """Trim off light-curve plateau to leave only the transient part +/- 50 time-steps

        Parameters
        ----------
        object_list: List[str]
            List of objects to apply the transformation to
        df: pd.DataFrame
            DataFrame containing the full light curve including dead points.

        Returns
        -------
        obs_transient, list(new_filtered_object_list): (pd.DataFrame, List[np.array])
            Tuple containing the updated dataframe with only the transient section, and a list of
            objects that the transformation was successful for. Note, some objects may cause an error
            and hence would not be returned in the new transformed dataframe

        Examples
        --------
        >>> object_list = list(np.unique(df["object_id"]))
        >>> obs_transient, object_list = __transient_trim(object_list, df)
        >>> generated_gp_dataset = generate_gp_all_objects(
            object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
            )
        ...
        """
        unixtimestamp = int(time.time())
        try:
            label = (
                subprocess.check_output(["git", "describe", "--always"])
                .strip()
                .decode()
            )
        except Exception:
            from astronet import __version__ as current_version

            label = current_version
        checkpoint_path = f"{asnwd}/astronet/t2/models/{self.dataset}/model-{os.environ.get('JOB_ID')}-{unixtimestamp}-{label}"
        csv_logger_file = f"{asnwd}/logs/t2/training-{os.environ.get('JOB_ID')}-{unixtimestamp}-{label}.log"

        if self.redshift is not None:
            X_train, y_train, X_test, y_test, loss, ZX_train, ZX_test = load_dataset(
                dataset=self.dataset,
                redshift=self.redshift,
                augmented=self.augmented,
                avocado=self.avocado,
                testset=self.testset,
            )
            hyper_results_file = (
                f"{asnwd}/astronet/t2/opt/runs/{self.dataset}/results_with_z.json"
            )
            num_aux_feats = ZX_train.shape[1]
        else:
            X_train, y_train, X_test, y_test, loss = load_dataset(
                dataset,
                augmented=self.augmented,
                avocado=self.avocado,
                testset=self.testset,
                fink=self.fink,
            )
            hyper_results_file = f"{asnwd}/astronet/t2/opt/runs/{dataset}/results.json"
            num_aux_feats = 0

        num_classes = y_train.shape[1]

        log.info(f"{X_train.shape, y_train.shape}")

        with open(hyper_results_file) as f:
            events = json.load(f)
            if self.model is not None:
                # Get params for model chosen with cli args
                event = next(
                    item
                    for item in events["optuna_result"]
                    if item["name"] == self.model
                )
            #            elif self.balance is not None:
            #                event = min(
            #                    (item for item in events["optuna_result"] if item["balanced_classes"] is not None),
            #                    key=lambda ev: ev["objective_score"],
            #                )
            else:
                # event = min(
                #     (item for item in events["optuna_result"] if item["balanced_classes"] is None),
                #     key=lambda ev: ev["objective_score"],
                # )
                event = min(
                    events["optuna_result"], key=lambda ev: ev["objective_score"]
                )

        embed_dim = event["embed_dim"]  # --> Embedding size for each token
        num_heads = event["num_heads"]  # --> Number of attention heads
        ff_dim = event[
            "ff_dim"
        ]  # --> Hidden layer size in feed forward network inside transformer

        # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
        num_filters = embed_dim

        num_layers = event["num_layers"]  # --> N x repeated transformer blocks
        droprate = event["droprate"]  # --> Rate of neurons to drop
        # fc_neurons = event['fc_neurons']    # --> N neurons in final Feed forward network.

        (
            num_samples,
            timesteps,
            num_features,
        ) = X_train.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)
        BATCH_SIZE = find_optimal_batch_size(num_samples)
        log.info(f"BATCH_SIZE:{BATCH_SIZE}")
        input_shape = (BATCH_SIZE, timesteps, num_features)
        log.info(f"input_shape:{input_shape}")

        VALIDATION_BATCH_SIZE = find_optimal_batch_size(X_test.shape[0])
        log.info(f"VALIDATION_BATCH_SIZE:{VALIDATION_BATCH_SIZE}")

        def get_compiled_model_and_data(loss):

            # input_shapes = (
            #     [input_shape, ZX_train.shape] if self.redshift is not None else input_shape
            # )
            if self.redshift is not None:
                input_shapes = [input_shape, ZX_train.shape]
                # model.build_graph(input_shapes)

                train_input = [X_train, ZX_train]
                test_input = [X_test, ZX_test]

                train_ds = (
                    tf.data.Dataset.from_tensor_slices(
                        (
                            {"input_1": train_input[0], "input_2": train_input[1]},
                            y_train,
                        )
                    )
                    .shuffle(1000)
                    .batch(BATCH_SIZE, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE)
                )
                test_ds = (
                    tf.data.Dataset.from_tensor_slices(
                        ({"input_1": test_input[0], "input_2": test_input[1]}, y_test)
                    )
                    .batch(BATCH_SIZE, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE)
                )
                # if avocado is not None:
                # Generate random boolean mask the length of data
                # use p 0.90 for False and 0.10 for True, i.e down-sample by 90%
                # mask = np.random.choice([False, True], len(X_test), p=[0.90, 0.10])
                # test_input = [X_test[mask], ZX_test[mask]]
                # y_test = y_test[mask]
            else:
                # model.build_graph(input_shape)
                input_shapes = input_shape
                train_input = X_train
                test_input = X_test

                train_ds = (
                    tf.data.Dataset.from_tensor_slices((train_input, y_train))
                    .shuffle(1000)
                    .batch(BATCH_SIZE, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE)
                )
                test_ds = (
                    tf.data.Dataset.from_tensor_slices((test_input, y_test))
                    .batch(BATCH_SIZE, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE)
                )

            model = build_model(
                input_shapes,
                input_dim=input_shape,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_filters=num_filters,
                num_classes=num_classes,
                num_layers=num_layers,
                droprate=droprate,
                num_aux_feats=num_aux_feats,
                add_aux_feats_to="L",
                # Either add features to M dimension or L dimension. Adding to L allows for
                # visualisation of CAMs relating to redshift since we would have a CAM of (L + Z) x c
                # fc_neurons=fc_neurons,
            )

            # We compile our model with a sampled learning rate and any custom metrics
            # lr = event["lr"]
            model.compile(
                loss=loss,
                optimizer=optimizers.Adam(lr=event["lr"], clipnorm=1),
                metrics=["acc"],
                run_eagerly=True,  # Show values when debugging. Also required for use with custom_log_loss
            )

            return model, train_ds, test_ds

        if len(tf.config.list_physical_devices("GPU")) > 1:
            # Create a MirroredStrategy.
            strategy = tf.distribute.MirroredStrategy()
            log.info("Number of devices: {}".format(strategy.num_replicas_in_sync))
            BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
            VALIDATION_BATCH_SIZE = (
                VALIDATION_BATCH_SIZE * strategy.num_replicas_in_sync
            )
            # Open a strategy scope.
            with strategy.scope():
                # If you are using a `Loss` class instead, set reduction to `NONE` so that
                # we can do the reduction afterwards and divide by global batch size.
                loss = DistributedWeightedLogLoss(
                    reduction=tf.keras.losses.Reduction.AUTO,
                    global_batch_size=BATCH_SIZE,
                )

                # Compute loss that is scaled by global batch size.
                # loss = tf.reduce_sum(loss_obj()) * (1.0 / BATCH_SIZE)

                # If clustering weights (model compression), build_model. Otherwise, T2Model should produce
                # original model. TODO: Include flag for choosing between the two, following run with FINK
                model, train_ds, test_ds = get_compiled_model_and_data(loss)
        else:
            model, train_ds, test_ds = get_compiled_model_and_data(loss)

        time_callback = TimeHistoryCallback()

        history = model.fit(
            train_ds,
            batch_size=BATCH_SIZE,
            epochs=self.epochs,
            shuffle=True,
            validation_data=test_ds,
            validation_batch_size=VALIDATION_BATCH_SIZE,
            verbose=False,
            callbacks=[
                time_callback,
                #                DetectOverfittingCallback(
                #                    threshold=2
                #                ),
                CSVLogger(
                    csv_logger_file,
                    separator=",",
                    append=False,
                ),
                EarlyStopping(
                    min_delta=0.001,
                    mode="min",
                    monitor="val_loss",
                    patience=50,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    mode="min",
                    monitor="val_loss",
                    save_best_only=True,
                ),
                ReduceLROnPlateau(
                    cooldown=5,
                    factor=0.1,
                    mode="min",
                    monitor="loss",
                    patience=5,
                    verbose=1,
                ),
            ],
        )

        model.summary(print_fn=logging.info)

        model.save(
            f"{asnwd}/astronet/t2/models/{self.dataset}/model-{os.environ.get('JOB_ID')}-{unixtimestamp}-{label}"
        )
        model.save_weights(
            f"{asnwd}/astronet/t2/models/{self.dataset}/weights-{os.environ.get('JOB_ID')}-{unixtimestamp}-{label}"
        )

        log.info(f"PER EPOCH TIMING: {time_callback.times}")
        log.info(f"AVERAGE EPOCH TIMING: {np.array(time_callback.times).mean()}")

        log.info(f"PERCENT OF RAM USED: {psutil.virtual_memory().percent}")
        log.info(f"RAM USED: {psutil.virtual_memory().active / (1024*1024*1024)}")

        #        with tf.device("/cpu:0"):
        #            try:
        #                print(f"LL-FULL Model Evaluate: {model.evaluate(test_input, y_test, verbose=0, batch_size=X_test.shape[0])[0]}")
        #            except Exception:
        #                print(f"Preventing possible OOM...")

        log.info(
            f"LL-BATCHED-32 Model Evaluate: {model.evaluate(test_ds, y_test, verbose=0)[0]}"
        )
        log.info(
            f"LL-BATCHED-OP Model Evaluate: {model.evaluate(test_ds, y_test, verbose=0, batch_size=VALIDATION_BATCH_SIZE)[0]}"
        )

        wloss = WeightedLogLoss()
        y_preds = model.predict(test_ds)
        log.info(f"LL-Test Model Predictions: {wloss(y_test, y_preds).numpy():.8f}")

        if X_test.shape[0] < 10000:
            batch_size = X_test.shape[0]  # Use all samples in test set to evaluate
        else:
            # Otherwise potential OOM Error may occur loading too many into memory at once
            batch_size = VALIDATION_BATCH_SIZE

        model_params = {}
        model_params["name"] = f"{os.environ.get('JOB_ID')}-{unixtimestamp}-{label}"
        model_params["hypername"] = event["name"]
        model_params["embed_dim"] = event["embed_dim"]
        model_params["ff_dim"] = event["ff_dim"]
        model_params["num_heads"] = event["num_heads"]
        model_params["num_layers"] = event["num_layers"]
        model_params["droprate"] = event["droprate"]
        # model_params['fc_neurons'] = event['fc_neurons']
        model_params["z-redshift"] = self.redshift
        model_params["augmented"] = self.augmented
        model_params["avocado"] = self.avocado
        model_params["testset"] = self.testset
        model_params["fink"] = self.fink
        model_params["num_classes"] = num_classes
        model_params["model_evaluate_on_test_acc"] = model.evaluate(
            test_ds, y_test, verbose=0, batch_size=batch_size
        )[1]
        model_params["model_evaluate_on_test_loss"] = model.evaluate(
            test_ds, y_test, verbose=0, batch_size=batch_size
        )[0]
        model_params["model_prediction_on_test"] = wloss(y_test, y_preds).numpy()

        y_test = np.argmax(y_test, axis=1)
        y_preds = np.argmax(y_preds, axis=1)

        model_params["model_predict_precision_score"] = precision_score(
            y_test, y_preds, average="macro"
        )
        model_params["model_predict_recall_score"] = recall_score(
            y_test, y_preds, average="macro"
        )

        print("  Params: ")
        for key, value in history.history.items():
            print("    {}: {}".format(key, value))
            model_params["{}".format(key)] = value

        del model_params["lr"]

        if self.redshift is not None:
            train_results_file = (
                f"{asnwd}/astronet/t2/models/{self.dataset}/results_with_z.json"
            )
        else:
            train_results_file = (
                f"{asnwd}/astronet/t2/models/{self.dataset}/results.json"
            )

        with open(train_results_file) as jf:
            data = json.load(jf)
            # print(data)

            previous_results = data["training_result"]
            # appending data to optuna_result
            # print(previous_results)
            previous_results.append(model_params)
            # print(previous_results)
            # print(data)

        with open(train_results_file, "w") as rf:
            json.dump(data, rf, sort_keys=True, indent=4)

        # PRUNE
        import tensorflow_model_optimization as tfmot

        # Helper function uses `prune_low_magnitude` to make only the
        # Dense layers train with pruning.
        def apply_pruning_to_dense(layer):
            if isinstance(layer, tf.keras.layers.Dense):
                return tfmot.sparsity.keras.prune_low_magnitude(layer)
            return layer

        # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense`
        # to the layers of the model.
        model_for_pruning = tf.keras.models.clone_model(
            model,
            clone_function=apply_pruning_to_dense,
        )

        model_for_pruning.summary(print_fn=logging.info)

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
        ]

        model_for_pruning.compile(
            loss=loss,
            optimizer=optimizers.Adam(lr=event["lr"], clipnorm=1),
            metrics=["acc"],
            run_eagerly=True,  # Show values when debugging. Also required for use with custom_log_loss
        )

        model_for_pruning.fit(
            train_ds,
            callbacks=callbacks,
            epochs=2,
        )

        model_for_pruning.save(
            f"{asnwd}/astronet/t2/models/{self.dataset}/model-{os.environ.get('JOB_ID')}-{unixtimestamp}-{label}-PRUNED",
            include_optimizer=True,
        )

        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        model_for_export.save(
            f"{asnwd}/astronet/t2/models/{self.dataset}/model-{os.environ.get('JOB_ID')}-{unixtimestamp}-{label}-EXPORT",
            include_optimizer=True,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process named model")

    parser.add_argument(
        "-d",
        "--dataset",
        default="wisdm_2010",
        help="Choose which dataset to use; options include: 'wisdm_2010', 'wisdm_2019'",
    )

    parser.add_argument(
        "-e", "--epochs", default=20, help="How many epochs to run training for"
    )

    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Name of tensorflow.keras model, i.e. model-<timestamp>-<hash>",
    )

    parser.add_argument(
        "-z",
        "--redshift",
        default=None,
        help="Whether to include redshift features or not",
    )

    parser.add_argument(
        "-a", "--augment", default=None, help="Train using augmented plasticc data"
    )

    parser.add_argument(
        "-A",
        "--avocado",
        default=None,
        help="Train using avocado augmented plasticc data",
    )

    parser.add_argument(
        "-t",
        "--testset",
        default=None,
        help="Train using PLAsTiCC test data for representative test",
    )

    parser.add_argument(
        "-f",
        "--fink",
        default=None,
        help="Train using PLAsTiCC but only g and r bands for FINK",
    )

    try:
        args = parser.parse_args()
        argsdict = vars(args)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    dataset = args.dataset
    EPOCHS = int(args.epochs)
    model = args.model

    augmented = args.augment
    if augmented is not None:
        augmented = True

    avocado = args.avocado
    if avocado is not None:
        avocado = True

    testset = args.testset
    if testset is not None:
        testset = True

    redshift = args.redshift
    if redshift is not None:
        redshift = True

    fink = args.fink
    if fink is not None:
        fink = True

    training = Training(
        epochs=EPOCHS,
        dataset=dataset,
        model=model,
        redshift=redshift,
        augmented=augmented,
        avocado=avocado,
        testset=testset,
        fink=fink,
    )
    if dataset in ["WalkvsRun", "NetFlow"]:
        # WalkvsRun and NetFlow causes OOM errors on GPU, run on CPU instead
        with tf.device("/cpu:0"):
            print(f"{dataset} causes OOM errors on GPU. Running on CPU...")
            training()
    else:
        print("Running on GPU...")
        training()
