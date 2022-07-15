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
import logging
import os
import shutil
import subprocess
import sys
import time
import warnings
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
from astronet.constants import SYSTEM
from astronet.custom_callbacks import (
    PrintModelSparsity,
    TimeHistoryCallback,
)
from astronet.datasets import (
    lazy_load_plasticc_noZ,
    lazy_load_plasticc_wZ,
)
from astronet.fetch_models import fetch_model
from astronet.metrics import (
    DistributedWeightedLogLoss,
    WeightedLogLoss,
)
from astronet.utils import astronet_logger, find_optimal_batch_size

try:
    log = astronet_logger(__file__)
    log.info("Running...\n" + "=" * (shutil.get_terminal_size((80, 20))[0]))
    log.info(f"File Path: {Path(__file__).absolute()}")
    log.info(f"Parent of Directory Path: {Path().absolute().parent}")
except Exception as e:
    print(f"{e}: Seems you are running from a notebook...")
    __file__ = f"{Path().resolve().parent}/astronet/train.py"
    log = astronet_logger(__file__)

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"


warnings.filterwarnings("ignore")


class Training(object):
    def __init__(
        self,
        epochs: int,
        dataset: str,
        model: str,
        redshift: bool,
        architecture: str,
        avocado: bool,
        testset: bool,
        fink: bool,
    ):
        self.architecture = architecture
        self.epochs = epochs
        self.dataset = dataset
        self.model = model
        self.redshift = redshift
        self.avocado = avocado
        self.testset = testset
        self.fink = fink

    def __call__(self):
        """Train a given architecture with, or without redshift, on either UGRIZY or GR passbands

        Parameters
        ----------
        epochs: int
            Number of epochs to run training for. If running locally, this should be < 5
        dataset: str
            Which dataset to train on; current options: {plasticc, wisdm_2010, wisdm_2019}
        model: str
            Model name of the best performing hyperparameters run
        redshift: bool
            Include additional information or redshift and redshift_error
        architecture: str
            Which architecture to train on; current options: {atx, t2, tinho}
        avocado: bool
            Run using augmented data generated from `avocado` pacakge
        testset: bool
            Run using homebrewed dataset constructed from PLAsTiCC 'test set'
        fink: bool
            Reduce number of bands from UGRIZY --> GR for ZTF like run.

        Examples
        --------
        >>> params = {
            "epochs": 2,
            "architecture": architecture,
            "dataset": dataset,
            "model": hyperrun,
            "testset": True,
            "redshift": True,
            "fink": None,
            "avocado": None,
        }
        >>> training = Training(**params)
        >>> loss = training.get_wloss
        """

        def build_label():
            UNIXTIMESTAMP = int(time.time())
            try:
                VERSION = (
                    subprocess.check_output(["git", "describe", "--always"])
                    .strip()
                    .decode()
                )
            except Exception:
                from astronet import __version__ as current_version

                VERSION = current_version
            JOB_ID = os.environ.get("JOB_ID")
            LABEL = f"{UNIXTIMESTAMP}-{JOB_ID}-{VERSION}"

            return LABEL

        LABEL = build_label()
        checkpoint_path = f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/checkpoints/checkpoint-{LABEL}"
        csv_logger_file = f"{asnwd}/logs/{self.architecture}/training-{LABEL}.log"

        # Lazy load data
        X_train = np.load(f"{asnwd}/data/plasticc/processed/X_train.npy", mmap_mode="r")
        Z_train = np.load(f"{asnwd}/data/plasticc/processed/Z_train.npy", mmap_mode="r")
        y_train = np.load(f"{asnwd}/data/plasticc/processed/y_train.npy", mmap_mode="r")

        X_test = np.load(f"{asnwd}/data/plasticc/processed/X_test.npy", mmap_mode="r")
        Z_test = np.load(f"{asnwd}/data/plasticc/processed/Z_test.npy", mmap_mode="r")
        y_test = np.load(f"{asnwd}/data/plasticc/processed/y_test.npy", mmap_mode="r")

        # >>> train_ds.element_spec[1].shape
        # TensorShape([14])
        # num_classes = train_ds.element_spec[1].shape.as_list()[0]
        num_classes = y_train.shape[1]

        if self.fink is not None:
            # Take only G, R bands
            X_train = X_train[:, :, 0:3:2]
            X_test = X_test[:, :, 0:3:2]

        log.info(f"{X_train.shape, y_train.shape}")

        (
            num_samples,
            timesteps,
            num_features,
        ) = X_train.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)

        BATCH_SIZE = find_optimal_batch_size(num_samples)
        log.info(f"BATCH_SIZE:{BATCH_SIZE}")

        input_shape = (BATCH_SIZE, timesteps, num_features)
        log.info(f"input_shape:{input_shape}")

        drop_remainder = False

        def get_compiled_model_and_data(loss, drop_remainder):

            if self.redshift is not None:
                hyper_results_file = f"{asnwd}/astronet/{self.architecture}/opt/runs/{self.dataset}/results_with_z.json"
                input_shapes = [input_shape, (BATCH_SIZE, Z_train.shape[1])]

                train_ds = (
                    lazy_load_plasticc_wZ(X_train, Z_train, y_train)
                    .shuffle(1000, seed=RANDOM_SEED)
                    .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                    .prefetch(tf.data.AUTOTUNE)
                    .cache()
                )
                test_ds = (
                    lazy_load_plasticc_wZ(X_test, Z_test, y_test)
                    .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                    .prefetch(tf.data.AUTOTUNE)
                    .cache()
                )

            else:
                hyper_results_file = f"{asnwd}/astronet/{self.architecture}/opt/runs/{self.dataset}/results.json"
                input_shapes = input_shape

                train_ds = (
                    lazy_load_plasticc_noZ(X_train, y_train)
                    .shuffle(1000, seed=RANDOM_SEED)
                    .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                    .prefetch(tf.data.AUTOTUNE)
                    .cache()
                )
                test_ds = (
                    lazy_load_plasticc_noZ(X_test, y_test)
                    .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                    .prefetch(tf.data.AUTOTUNE)
                    .cache()
                )

            model, event = fetch_model(
                model=self.model,
                hyper_results_file=hyper_results_file,
                input_shapes=input_shapes,
                architecture=self.architecture,
                num_classes=num_classes,
            )

            # We compile our model with a sampled learning rate and any custom metrics
            learning_rate = event["lr"]
            model.compile(
                loss=loss,
                optimizer=optimizers.Adam(learning_rate=learning_rate, clipnorm=1),
                metrics=["acc"],
                run_eagerly=True,  # Show values when debugging. Also required for use with custom_log_loss
            )

            return model, train_ds, test_ds, event, hyper_results_file

        VALIDATION_BATCH_SIZE = find_optimal_batch_size(X_test.shape[0])
        log.info(f"VALIDATION_BATCH_SIZE:{VALIDATION_BATCH_SIZE}")

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
                    # global_batch_size=BATCH_SIZE,
                )

                # Compute loss that is scaled by global batch size.
                # loss = tf.reduce_sum(loss_obj()) * (1.0 / BATCH_SIZE)

                # If clustering weights (model compression), build_model. Otherwise, T2Model should produce
                # original model. TODO: Include flag for choosing between the two, following run with FINK
                (
                    model,
                    train_ds,
                    test_ds,
                    event,
                    hyper_results_file,
                ) = get_compiled_model_and_data(loss, drop_remainder)
        else:
            loss = WeightedLogLoss()
            (
                model,
                train_ds,
                test_ds,
                event,
                hyper_results_file,
            ) = get_compiled_model_and_data(loss, drop_remainder)

        if "pytest" in sys.modules or SYSTEM == "Darwin":
            NTAKE = 3

            train_ds = train_ds.take(NTAKE)
            test_ds = test_ds.take(NTAKE)

            ind = np.array([x for x in range(NTAKE * BATCH_SIZE)])
            y_test = np.take(y_test, ind, axis=0)

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
            f"LL-BATCHED-32 Model Evaluate: {model.evaluate(test_ds, verbose=0)[0]}"
        )
        log.info(
            f"LL-BATCHED-OP Model Evaluate: {model.evaluate(test_ds, verbose=0, batch_size=VALIDATION_BATCH_SIZE)[0]}"
        )

        if drop_remainder:
            ind = np.array(
                [x for x in range((y_test.shape[0] // BATCH_SIZE) * BATCH_SIZE)]
            )
            y_test = np.take(y_test, ind, axis=0)

        y_preds = model.predict(test_ds)

        log.info(f"{y_preds.shape}, {type(y_preds)}")

        WLOSS = loss(y_test, y_preds).numpy()
        log.info(f"LL-Test Model Predictions: {WLOSS:.8f}")
        if "pytest" in sys.modules:
            return WLOSS

        LABEL = (
            "wZ-" + LABEL if self.redshift else "noZ-" + LABEL
        )  # Prepend whether redshift was used or not
        LABEL = (
            "GR-" + LABEL if self.fink else "UGRIZY-" + LABEL
        )  # Prepend which filters have been used in training
        LABEL += f"-LL{WLOSS:.3f}"  # Append loss score

        if SYSTEM != "Darwin":
            model.save(
                f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/model-{LABEL}"
            )
            model.save_weights(
                f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/weights/weights-{LABEL}"
            )

        if X_test.shape[0] < 10000:
            batch_size = X_test.shape[0]  # Use all samples in test set to evaluate
        else:
            # Otherwise potential OOM Error may occur loading too many into memory at once
            batch_size = (
                int(VALIDATION_BATCH_SIZE / strategy.num_replicas_in_sync)
                if len(tf.config.list_physical_devices("GPU")) > 1
                else VALIDATION_BATCH_SIZE
            )
            log.info(f"EVALUATE VALIDATION_BATCH_SIZE : {batch_size}")

        event["hypername"] = event["name"]
        event["name"] = f"{LABEL}"

        event["z-redshift"] = self.redshift
        event["avocado"] = self.avocado
        event["testset"] = self.testset
        event["fink"] = self.fink

        event["num_classes"] = num_classes
        event["model_evaluate_on_test_acc"] = model.evaluate(
            test_ds, verbose=0, batch_size=batch_size
        )[1]
        event["model_evaluate_on_test_loss"] = model.evaluate(
            test_ds, verbose=0, batch_size=batch_size
        )[0]
        event["model_prediction_on_test"] = loss(y_test, y_preds).numpy()

        y_test = np.argmax(y_test, axis=1)
        y_preds = np.argmax(y_preds, axis=1)

        event["model_predict_precision_score"] = precision_score(
            y_test, y_preds, average="macro"
        )
        event["model_predict_recall_score"] = recall_score(
            y_test, y_preds, average="macro"
        )

        print("  Params: ")
        for key, value in history.history.items():
            print("    {}: {}".format(key, value))
            event["{}".format(key)] = value

        learning_rate = event["lr"]
        del event["lr"]

        if self.redshift is not None:
            train_results_file = f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/results_with_z.json"
        else:
            train_results_file = f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/results.json"

        with open(train_results_file) as jf:
            data = json.load(jf)
            # print(data)

            previous_results = data["training_result"]
            # appending data to optuna_result
            # print(previous_results)
            previous_results.append(event)
            # print(previous_results)
            # print(data)

        if SYSTEM != "Darwin":
            with open(train_results_file, "w") as rf:
                json.dump(data, rf, sort_keys=True, indent=4)

        if len(tf.config.list_physical_devices("GPU")) < 2 and SYSTEM != "Darwin":
            # PRUNE
            import tensorflow_model_optimization as tfmot

            # Helper function uses `prune_low_magnitude` to make only the
            # Dense layers train with pruning.
            def apply_pruning_to_dense(layer):
                layer_name = layer.__class__.__name__
                # prunable_layers = ["ConvEmbedding", "TransformerBlock", "ClusterWeights"]
                # if layer_name in prunable_layers:
                if isinstance(layer, tfmot.sparsity.keras.PrunableLayer):
                    log.info(f"Pruning {layer_name}")
                    return tfmot.sparsity.keras.prune_low_magnitude(layer)
                return layer

            # Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense`
            # to the layers of the model.
            model_for_pruning = tf.keras.models.clone_model(
                model,
                clone_function=apply_pruning_to_dense,
            )

            model_for_pruning.summary(print_fn=log.info)

            log_dir = f"{asnwd}/logs/{self.architecture}"

            callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir),
                PrintModelSparsity(),
                EarlyStopping(
                    min_delta=0.001,
                    mode="min",
                    monitor="val_loss",
                    patience=25,
                    restore_best_weights=True,
                    verbose=1,
                ),
            ]

            model_for_pruning.compile(
                loss=loss,
                optimizer=optimizers.Adam(learning_rate=learning_rate, clipnorm=1),
                metrics=["acc"],
                run_eagerly=True,  # Show values when debugging. Also required for use with custom_log_loss
            )

            model_for_pruning.fit(
                train_ds,
                callbacks=callbacks,
                epochs=100,
            )

            model_for_pruning.save(
                f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/model-{LABEL}-PRUNED",
                include_optimizer=True,
            )

            model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
            model_for_export.save(
                f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/model-{LABEL}-PRUNED-STRIPPED",
                include_optimizer=True,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process named model")

    parser.add_argument(
        "-a", "--architecture", default="tinho", help="Which architecture to train on"
    )

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

    architecture = args.architecture
    dataset = args.dataset
    EPOCHS = int(args.epochs)
    model = args.model

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
        architecture=architecture,
        dataset=dataset,
        model=model,
        redshift=redshift,
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
