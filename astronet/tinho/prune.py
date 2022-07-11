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
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import optimizers

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.custom_callbacks import PrintModelSparsity
from astronet.datasets import (
    lazy_load_plasticc_noZ,
    lazy_load_plasticc_wZ,
)
from astronet.get_models import get_clustered_model
from astronet.metrics import WeightedLogLoss
from astronet.utils import astronet_logger, find_optimal_batch_size

try:
    log = astronet_logger(__file__)
    log.info("\n" + "=" * (shutil.get_terminal_size((80, 20))[0]))
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


class Prune(object):
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

        def get_compiled_model_and_data(loss):

            if self.redshift is not None:
                hyper_results_file = f"{asnwd}/astronet/{self.architecture}/opt/runs/{self.dataset}/results_with_z.json"

                train_ds = (
                    lazy_load_plasticc_wZ(X_train, Z_train, y_train)
                    .shuffle(1000, seed=RANDOM_SEED)
                    .batch(BATCH_SIZE, drop_remainder=False)
                    .prefetch(tf.data.AUTOTUNE)
                )
                test_ds = (
                    lazy_load_plasticc_wZ(X_test, Z_test, y_test)
                    .batch(BATCH_SIZE, drop_remainder=False)
                    .prefetch(tf.data.AUTOTUNE)
                )

            else:
                hyper_results_file = f"{asnwd}/astronet/{self.architecture}/opt/runs/{self.dataset}/results.json"

                train_ds = (
                    lazy_load_plasticc_noZ(X_train, y_train)
                    .shuffle(1000, seed=RANDOM_SEED)
                    .batch(BATCH_SIZE, drop_remainder=False)
                    .prefetch(tf.data.AUTOTUNE)
                )
                test_ds = (
                    lazy_load_plasticc_noZ(X_test, y_test)
                    .batch(BATCH_SIZE, drop_remainder=False)
                    .prefetch(tf.data.AUTOTUNE)
                )

            model = get_clustered_model()

            return model, train_ds, test_ds, hyper_results_file

        VALIDATION_BATCH_SIZE = find_optimal_batch_size(X_test.shape[0])
        log.info(f"VALIDATION_BATCH_SIZE:{VALIDATION_BATCH_SIZE}")

        loss = WeightedLogLoss()
        (
            model,
            train_ds,
            test_ds,
            hyper_results_file,
        ) = get_compiled_model_and_data(loss)

        y_preds = model.predict(test_ds)
        WLOSS = loss(y_test, y_preds).numpy()
        log.info(f"Before Pruning LOSS: {WLOSS:.8f}")
        log.info(f"FINE-TUNING WITH {self.epochs} EPOCHS")

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
        ]

        learning_rate = 0.017
        model_for_pruning.compile(
            loss=loss,
            optimizer=optimizers.Adam(learning_rate=learning_rate, clipnorm=1),
            metrics=["acc"],
            run_eagerly=True,  # Show values when debugging. Also required for use with custom_log_loss
        )

        model_for_pruning.fit(
            train_ds,
            callbacks=callbacks,
            epochs=self.epochs,
        )

        y_preds = model_for_pruning.predict(test_ds)
        WLOSS = loss(y_test, y_preds).numpy()
        log.info(f"After Pruning LOSS: {WLOSS:.8f}")

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

    pruning = Prune(
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
            pruning()
    else:
        print("Running on GPU...")
        pruning()
