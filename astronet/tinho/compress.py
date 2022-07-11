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
import pathlib
import random as python_random
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from pandas.core.common import flatten
from tensorflow import keras

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.metrics import WeightedLogLoss
from astronet.utils import get_encoding

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)


def print_sparsity(model):
    for w in model.weights:
        n_weights = w.numpy().size
        n_zeros = np.count_nonzero(w == 0)
        sparsity = n_zeros / n_weights * 100.0
        if sparsity > 0:
            return f"{sparsity:.1f}% sparsity in {w.name} layer"
    return "No zero magnitude weights"


def print_clusters(model):
    for w in model.weights:
        n_weights = w.numpy().size
        n_unique = len(np.unique(w))
        if n_unique < n_weights:
            return f"{w.name} - {n_unique} unique weights, of {n_weights} total for given layer"


def inspect_model(model):
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    for name, weight in zip(names, weights):
        print(name, weight.shape)


class Compress(object):
    # TODO: Update docstrings
    def __init__(self, architecture, dataset, model_name, redshift, savefigs=True):
        self.architecture = architecture
        self.dataset = dataset
        self.model_name = model_name
        self.redshift = redshift

    def __call__(self):
        X_test = np.load(
            f"{asnwd}/data/plasticc/test_set/infer/X_test.npy",
        )
        y_test = np.load(
            f"{asnwd}/data/plasticc/test_set/infer/y_test.npy",
        )
        Z_test = np.load(
            f"{asnwd}/data/plasticc/test_set/infer/Z_test.npy",
        )

        print(f"X_TEST: {X_test.shape}, Y_TEST: {y_test.shape}, Z_TEST: {Z_test.shape}")

        dataform = "testset"
        encoding, class_encoding, class_names = get_encoding(
            self.dataset, dataform=dataform
        )

        y_true_test = encoding.inverse_transform(y_test)
        print("N_TEST:", Counter(list(flatten(y_true_test))))

        def check_size(filepath):
            du = subprocess.run(
                f"du -sh {filepath} | awk '{{print $1}}'",
                check=True,
                capture_output=True,
                shell=True,
                text=True,
            ).stdout
            return du

        def zippify(filepath, name):

            directory = pathlib.Path(filepath)

            zipped_name = f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/tinho/compressed_{name}.zip"

            with zipfile.ZipFile(
                zipped_name,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive:
                for file_path in directory.rglob("*"):
                    archive.write(file_path, arcname=file_path.relative_to(directory))

            return zipped_name

        def run_predictions_lsst(X_test, Z_test, wloss):
            # ORIGINAL
            original_model_fp = f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/model-{self.model_name}"
            original_model = keras.models.load_model(
                original_model_fp,
                custom_objects={"WeightedLogLoss": WeightedLogLoss()},
                compile=False,
            )
            inspect_model(original_model)
            print(f"ORIGINAL MODEL ON DISK: {check_size(original_model_fp)}")
            original_model_zipped = zippify(original_model_fp, "original_model")
            print(
                f"COMPRESSED ORIGINAL MODEL ON DISK: {check_size(original_model_zipped)}"
            )

            # CLUSTERED
            clustered_model_fp = f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/model-9902350-1652645235-0.5.1.dev14+gef9460b"

            clustered_model = keras.models.load_model(
                clustered_model_fp,
                custom_objects={"WeightedLogLoss": WeightedLogLoss()},
                compile=False,
            )
            inspect_model(clustered_model)
            print(f"CLUSTERED-STRIPPED MODEL ON DISK: {check_size(clustered_model_fp)}")

            # CLUSTERED-STRIPPED
            clustered_stripped_model = tfmot.clustering.keras.strip_clustering(
                clustered_model
            )
            clustered_stripped_model_fp = f"{Path(__file__).parent}/models/{self.dataset}/tinho/clustered_stripped_model"
            clustered_stripped_model.save(clustered_stripped_model_fp)
            clustered_stripped_model_zipped = zippify(
                clustered_stripped_model_fp, "clustered_stripped_model"
            )
            print(
                f"CLUSTERED-STRIPPED MODEL ON DISK: {check_size(clustered_stripped_model_fp)}"
            )
            print(
                f"COMPRESSED CLUSTERED-STRIPPED MODEL ON DISK: {check_size(clustered_stripped_model_zipped)}"
            )

            # PRUNE
            # pruned_model_fp = f"{Path(__file__).parent}/models/plasticc/model-9903403-1652652371-0.5.1.dev19+g23d6486.d20220515-PRUNED"
            # pruned_model = keras.models.load_model(
            #     pruned_model_fp,
            #     custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            #     compile=False,
            # )
            # print(f"PRUNED MODEL ON DISK: {check_size(pruned_model_fp)}")

            pruned_stripped_model_fp = f"{Path(__file__).parent}/models/plasticc/model-9903403-1652652371-0.5.1.dev19+g23d6486.d20220515-EXPORT"
            pruned_stripped_model = keras.models.load_model(
                pruned_stripped_model_fp,
                custom_objects={"WeightedLogLoss": WeightedLogLoss()},
                compile=False,
            )

            pruned_stripped_model_zipped = zippify(
                pruned_stripped_model_fp,
                f"{Path(__file__).parent}/models/{self.dataset}/tinho/pruned_stripped_model",
            )
            print(
                f"COMPRESSED PRUNED MODEL ON DISK: {check_size(pruned_stripped_model_zipped)}"
            )

            # ORIGINAL MODEL
            print("ORIGINAL MODEL LOSS")
            y_preds = original_model.predict([X_test, Z_test])
            print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")

            # CLUSTERED MODEL
            print("CLUSTERED-STRIPPED MODEL LOSS")
            y_preds = clustered_model.predict([X_test, Z_test])
            print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")

            # CLUSTERED-STRIPPED MODEL
            print("CLUSTERED-STRIPPED MODEL LOSS")
            y_preds = clustered_stripped_model.predict([X_test, Z_test])
            print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")

            # PRUNED-STRIPPED MODEL
            print("PRUNED-STRIPPED MODEL LOSS")
            y_preds = pruned_stripped_model.predict([X_test, Z_test])
            print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")

            print_sparsity(original_model)
            print_sparsity(clustered_stripped_model)
            print_sparsity(pruned_stripped_model)

            print_clusters(original_model)
            print_clusters(clustered_stripped_model)
            print_clusters(pruned_stripped_model)

        def run_predictions_ztf(X_test, wloss):

            # Only trained on red, green filters {r, g}
            X_test = X_test[:, :, 0:3:2]

            # ORIGINAL FINK MODEL
            original_fink_model_fp = f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/model-23057-1642540624-0.1.dev963+g309c9d8"
            original_fink_model = keras.models.load_model(
                original_fink_model_fp,
                custom_objects={"WeightedLogLoss": WeightedLogLoss()},
                compile=False,
            )
            inspect_model(original_fink_model)
            print(f"ORIGINAL FINK MODEL ON DISK: {check_size(original_fink_model_fp)}")
            original_fink_model_zipped = zippify(
                original_fink_model_fp, "original_fink_model"
            )
            print(
                f"COMPRESSED ORIGINAL FINK MODEL ON DISK: {check_size(original_fink_model_zipped)}"
            )

            print("ORIGINAL FINK MODEL LOSS")
            y_preds = original_fink_model.predict(X_test)
            print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")

            # CLUSTERED-STRIPPED FINK MODEL
            clustered_fink_model = keras.models.load_model(
                f"{asnwd}/astronet/{self.architecture}/models/plasticc/model-9903651-1652692724-0.5.1.dev24+gb7cd783.d20220516",
                custom_objects={"WeightedLogLoss": WeightedLogLoss()},
                compile=False,
            )
            inspect_model(clustered_fink_model)
            clustered_stripped_fink_model = tfmot.clustering.keras.strip_clustering(
                clustered_fink_model
            )

            clustered_stripped_fink_model_fp = f"{Path(__file__).parent}/models/{self.dataset}/tinho/clustered_stripped_fink_model"
            clustered_stripped_fink_model.save(clustered_stripped_fink_model_fp)
            clustered_stripped_fink_model_zipped = zippify(
                clustered_stripped_fink_model_fp, "clustered_stripped_fink_model"
            )
            print(
                f"CLUSTERED-STRIPPED FINK MODEL ON DISK: {check_size(clustered_stripped_fink_model_fp)}"
            )
            print(
                f"COMPRESSED CLUSTERED-STRIPPED FINK MODEL ON DISK: {check_size(clustered_stripped_fink_model_zipped)}"
            )

            print("CLUSTERED-STRIPPED FINK MODEL LOSS")
            y_preds = clustered_stripped_fink_model.predict(X_test)
            print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")

            # PRUNED FINK MODEL
            pruned_fink_model_fp = f"{asnwd}/astronet/{self.architecture}/models/plasticc/model-9903651-1652692724-0.5.1.dev24+gb7cd783.d20220516-PRUNED"
            pruned_fink_model = keras.models.load_model(
                pruned_fink_model_fp,
                custom_objects={"WeightedLogLoss": WeightedLogLoss()},
                compile=False,
            )
            inspect_model(pruned_fink_model)
            print(f"PRUNED FINK MODEL ON DISK: {check_size(pruned_fink_model_fp)}")
            # pruned_fink_model_zipped = zippify(pruned_fink_model_fp, "pruned_fink_model")
            # print(f"COMPRESSED PRUNED FINK MODEL ON DISK: {check_size(pruned_fink_model_zipped)}")

            # print("PRUNED FINK MODEL LOSS")
            # y_preds = pruned_stripped_fink_model.predict(X_test)
            # print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")

            # PRUNED-STRIPPED FINK MODEL
            pruned_stripped_fink_model_fp = f"{asnwd}/astronet/{self.architecture}/models/plasticc/model-9903651-1652692724-0.5.1.dev24+gb7cd783.d20220516-EXPORT"
            pruned_stripped_fink_model = keras.models.load_model(
                pruned_stripped_fink_model_fp,
                custom_objects={"WeightedLogLoss": WeightedLogLoss()},
                compile=False,
            )
            inspect_model(pruned_stripped_fink_model)
            print(
                f"PRUNED-STRIPPED FINK MODEL ON DISK: {check_size(pruned_stripped_fink_model_fp)}"
            )
            pruned_stripped_fink_model_zipped = zippify(
                pruned_fink_model_fp, "pruned_stripped_fink_model"
            )
            print(
                f"COMPRESSED PRUNED-STRIPPED FINK MODEL ON DISK: {check_size(pruned_stripped_fink_model_zipped)}"
            )

            print("PRUNED-STRIPPED FINK MODEL LOSS")
            y_preds = pruned_stripped_fink_model.predict(X_test)
            print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")

            print_sparsity(original_fink_model)
            print_sparsity(clustered_stripped_fink_model)
            print_sparsity(pruned_stripped_fink_model)

            print_clusters(original_fink_model)
            print_clusters(clustered_stripped_fink_model)
            print_clusters(pruned_stripped_fink_model)

        print("Running predictions")
        wloss = WeightedLogLoss()

        # run_predictions_lsst(X_test, Z_test, wloss)
        run_predictions_ztf(X_test, wloss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Playground for testing t2 compression"
    )

    parser.add_argument(
        "-a",
        "--architecture",
        default="t2",
        help="Choose which architecture to evaluate : 'atx', 't2'",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        default="plasticc",
        help="Choose which dataset to use: This is fixed for plasticc for now",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="1619624444-0.1.dev765+g7c90cbb.d20210428",
        help="Name of tensorflow.keras model, i.e. model-<timestamp>-<hash>",
    )

    parser.add_argument(
        "-z",
        "--redshift",
        default=None,
        help="Whether to include redshift features or not",
    )

    try:
        args = parser.parse_args()
        argsdict = vars(args)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    architecture = args.architecture
    dataset = args.dataset
    model_name = args.model
    redshift = args.redshift
    if redshift is not None:
        redshift = True

    compression = Compress(
        architecture=architecture,
        dataset=dataset,
        model_name=model_name,
        redshift=redshift,
    )
    compression()
