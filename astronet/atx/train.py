# Copyright 2020 - 2022
# Author: Tarek Allam Jr.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
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

from astronet.atx.model import ATXModel
from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.custom_callbacks import (
    DetectOverfittingCallback,
    SGEBreakoutCallback,
    TimeHistoryCallback,
)
from astronet.metrics import WeightedLogLoss
from astronet.utils import (
    astronet_logger,
    find_optimal_batch_size,
    get_data_count,
    load_dataset,
)

try:
    log = astronet_logger(__file__)
    log.info("=" * shutil.get_terminal_size((80, 20))[0])
    log.info(f"File Path: {Path(__file__).absolute()}")
    log.info(f"Parent of Directory Path: {Path().absolute().parent}")
except Exception as e:
    print(f"{e}: Seems you are running from a notebook...")
    __file__ = f"{Path().resolve().parent}/astronet/atx/train.py"
    log = astronet_logger(__file__)

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"


class Training(object):
    def __init__(
        self, epochs, dataset, model, redshift, balance, avocado, testset, fink
    ):
        self.epochs = epochs
        self.dataset = dataset
        self.model = model
        self.redshift = redshift
        self.balance = balance
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

        if self.redshift is not None:
            X_train, y_train, X_test, y_test, loss, ZX_train, ZX_test = load_dataset(
                dataset=self.dataset,
                redshift=self.redshift,
                balance=self.balance,
                avocado=self.avocado,
                testset=self.testset,
            )
            hyper_results_file = (
                f"{asnwd}/astronet/atx/opt/runs/{self.dataset}/results_with_z.json"
            )
        else:
            X_train, y_train, X_test, y_test, loss = load_dataset(
                dataset=self.dataset,
                balance=self.balance,
                avocado=self.avocado,
                testset=self.testset,
                fink=self.fink,
            )
            hyper_results_file = f"{asnwd}/astronet/atx/opt/runs/{dataset}/results.json"

        num_classes = y_train.shape[1]

        y_train_count, y_test_count = get_data_count(
            dataset=dataset, dataform=testset, y_train=y_train, y_test=y_test
        )
        log.info(f"{X_train.shape, y_train.shape}")
        log.info(f"N-TRAIN: {y_train_count}")
        log.info(f"N-TEST: {y_test_count}")

        with open(hyper_results_file) as f:
            events = json.load(f)
            if self.model is not None:
                # Get params for model chosen with cli args
                event = next(
                    item
                    for item in events["optuna_result"]
                    if item["name"] == self.model
                )
            else:
                event = min(
                    events["optuna_result"], key=lambda ev: ev["objective_score"]
                )

        kernel_size = event["kernel_size"]  # --> Filter length
        pool_size = event["pool_size"]  # --> Pooling width
        scaledown_factor = event[
            "scaledown_factor"
        ]  # --> Reduce number of filters down by given factor

        (
            num_samples,
            timesteps,
            num_features,
        ) = X_train.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)
        BATCH_SIZE = find_optimal_batch_size(num_samples)
        print(f"BATCH_SIZE:{BATCH_SIZE}")
        input_shape = (BATCH_SIZE, timesteps, num_features)
        print(f"input_shape:{input_shape}")

        VALIDATION_BATCH_SIZE = find_optimal_batch_size(X_test.shape[0])
        print(f"VALIDATION_BATCH_SIZE:{VALIDATION_BATCH_SIZE}")

        def get_saved_model():
            model = tf.keras.models.load_model(
                f"{asnwd}/astronet/atx/models/{dataset}/model-9874103-1640950366-0.1.dev932+g9422fe9",
                custom_objects={"WeightedLogLoss": WeightedLogLoss()},
                compile=False,
            )

            # We compile our model with a sampled learning rate and any custom metrics
            lr = event["lr"]
            model.compile(
                loss=loss,
                optimizer=optimizers.Adam(lr=lr, clipnorm=1),
                metrics=["acc"],
                run_eagerly=True,
                # True for values when debugging. Also required for use with custom_log_loss
                # Also prevents NotImplementedError: Cannot convert a symbolic Tensor
                # (cond_2/Identity_1:0) to a numpy array. This error may indicate that you're trying
                # to pass a Tensor to a NumPy call, which is not supported
            )

            if self.redshift is not None:
                train_input = [X_train, ZX_train]
                test_input = [X_test, ZX_test]
            else:
                train_input = X_train
                test_input = X_test

            log.info("Loading from saved model...")
            return model, train_input, test_input

        def get_compiled_model():
            model = ATXModel(
                num_classes=num_classes,
                kernel_size=kernel_size,
                pool_size=pool_size,
                scaledown_factor=scaledown_factor,
            )

            # We compile our model with a sampled learning rate and any custom metrics
            lr = event["lr"]
            model.compile(
                loss=loss,
                optimizer=optimizers.Adam(lr=lr, clipnorm=1),
                metrics=["acc"],
                run_eagerly=True,
                # True for values when debugging. Also required for use with custom_log_loss
                # Also prevents NotImplementedError: Cannot convert a symbolic Tensor
                # (cond_2/Identity_1:0) to a numpy array. This error may indicate that you're trying
                # to pass a Tensor to a NumPy call, which is not supported
            )

            if self.redshift is not None:
                input_shapes = [input_shape, ZX_train.shape]
                model.build_graph(input_shapes)

                train_input = [X_train, ZX_train]
                test_input = [X_test, ZX_test]
                # if avocado is not None:
                # Generate random boolean mask the length of data
                # use p 0.90 for False and 0.10 for True, i.e down-sample by 90%
                # mask = np.random.choice([False, True], len(X_test), p=[0.90, 0.10])
                # test_input = [X_test[mask], ZX_test[mask]]
                # y_test = y_test[mask]
            else:
                model.build_graph(input_shape)

                train_input = X_train
                test_input = X_test

            log.info("New atx model compiled...")
            return model, train_input, test_input

        if len(tf.config.list_physical_devices("GPU")) > 1:
            # Create a MirroredStrategy.
            strategy = tf.distribute.MirroredStrategy()
            print("Number of devices: {}".format(strategy.num_replicas_in_sync))
            # Open a strategy scope.
            with strategy.scope():
                model, train_input, test_input = get_compiled_model()
        else:
            model, train_input, test_input = get_compiled_model()
            # model, train_input, test_input = get_saved_model()

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
        checkpoint_path = f"{asnwd}/astronet/atx/models/{self.dataset}/model-{os.environ.get('JOB_ID')}-{unixtimestamp}-{label}"
        csv_logger_file = f"{asnwd}/logs/atx/training-{os.environ.get('JOB_ID')}-{unixtimestamp}-{label}.log"

        time_callback = TimeHistoryCallback()

        history = model.fit(
            train_input,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=self.epochs,
            shuffle=True,
            validation_data=(test_input, y_test),
            validation_batch_size=VALIDATION_BATCH_SIZE,
            verbose=False,
            callbacks=[
                time_callback,
                #                SGEBreakoutCallback(
                #                    threshold=44  # Stop training if running for more than threshold number of hours
                # ),
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

        model.summary(print_fn=log.info)

        model.save(
            f"{asnwd}/astronet/atx/models/{self.dataset}/model-{os.environ.get('JOB_ID')}-{unixtimestamp}-{label}"
        )
        model.save_weights(
            f"{asnwd}/astronet/atx/models/{self.dataset}/weights-{os.environ.get('JOB_ID')}-{unixtimestamp}-{label}"
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

        print(
            f"LL-BATCHED-32 Model Evaluate: {model.evaluate(test_input, y_test, verbose=0)[0]}"
        )
        print(
            f"LL-BATCHED-OP Model Evaluate: {model.evaluate(test_input, y_test, verbose=0, batch_size=VALIDATION_BATCH_SIZE)[0]}"
        )

        wloss = WeightedLogLoss()
        y_preds = model.predict(test_input)
        print(f"LL-Test Model Predictions: {wloss(y_test, y_preds).numpy():.8f}")

        if X_test.shape[0] < 10000:
            batch_size = X_test.shape[0]  # Use all samples in test set to evaluate
        else:
            # Otherwise potential OOM Error may occur loading too many into memory at once
            batch_size = VALIDATION_BATCH_SIZE

        model_params = {}
        model_params["name"] = f"{os.environ.get('JOB_ID')}-{unixtimestamp}-{label}"
        model_params["hypername"] = event["name"]
        model_params["kernel_size"] = event["kernel_size"]
        model_params["pool_size"] = event["pool_size"]
        model_params["scaledown_factor"] = event["scaledown_factor"]

        model_params["z-redshift"] = self.redshift
        model_params["balance"] = self.balance
        model_params["avocado"] = self.avocado
        model_params["testset"] = self.testset
        model_params["fink"] = self.fink
        model_params["num_classes"] = num_classes
        model_params["model_evaluate_on_test_acc"] = model.evaluate(
            test_input, y_test, verbose=0, batch_size=batch_size
        )[1]
        model_params["model_evaluate_on_test_loss"] = model.evaluate(
            test_input, y_test, verbose=0, batch_size=batch_size
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
                f"{asnwd}/astronet/atx/models/{self.dataset}/results_with_z.json"
            )
        else:
            train_results_file = (
                f"{asnwd}/astronet/atx/models/{self.dataset}/results.json"
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
        "-b",
        "--balance",
        default=None,
        help="Train using balanced classes with augmented plasticc data",
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
        print(f"ARGS: {argsdict}")
    except KeyError:
        parser.print_help()
        sys.exit(0)

    dataset = args.dataset
    EPOCHS = int(args.epochs)
    model = args.model

    balance = args.balance
    if balance is not None:
        balance = True

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
        balance=balance,
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
