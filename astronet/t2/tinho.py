import argparse
import json
import os
import random as python_random
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import snappy
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.metrics import WeightedLogLoss
from astronet.utils import find_optimal_batch_size, get_encoding

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)


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

        (
            num_samples,
            timesteps,
            num_features,
        ) = X_test.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)
        BATCH_SIZE = find_optimal_batch_size(num_samples)
        print(f"BATCH_SIZE:{BATCH_SIZE}")

        if self.redshift is not None:
            # inputs = [X_test, Z_test]
            results_filename = f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/results_with_z.json"
        else:
            # inputs = X_test
            results_filename = f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/results.json"

        with open(results_filename) as f:
            events = json.load(f)
            if self.model_name is not None:
                # Get params for model chosen with cli args
                event = next(
                    item
                    for item in events["training_result"]
                    if item["name"] == self.model_name
                )
            else:
                event = min(
                    events["training_result"],
                    key=lambda ev: ev["model_evaluate_on_test_loss"],
                )

        dataform = "testset"
        encoding, class_encoding, class_names = get_encoding(
            self.dataset, dataform=dataform
        )

        from collections import Counter

        from pandas.core.common import flatten

        y_true_test = encoding.inverse_transform(y_test)
        print("N_TEST:", Counter(list(flatten(y_true_test))))

        logloss = event["model_evaluate_on_test_loss"]
        acc = event["model_evaluate_on_test_acc"]
        print(f"LogLoss on Test Set: {logloss}, Accuracy on Test Set: {acc}")

        model = keras.models.load_model(
            f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/model-{self.model_name}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

        names = [weight.name for layer in model.layers for weight in layer.weights]
        weights = model.get_weights()

        for name, weight in zip(names, weights):
            print(name, weight.shape)

        final_model = tfmot.clustering.keras.strip_clustering(model)

        print("final model")
        final_model.summary()

        original_model = keras.models.load_model(
            f"{asnwd}/astronet/{self.architecture}/models/{self.dataset}/model-1619624444-0.1.dev765+g7c90cbb.d20210428",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

        print("Running predictions")
        wloss = WeightedLogLoss()

        # ORIGINAL MODEL
        y_preds = original_model.predict([X_test, Z_test])
        print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")
        y_preds = original_model.predict([X_test, Z_test], batch_size=BATCH_SIZE)
        print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")

        # FINAL STRIPPED-CLUSTERED MODEL
        y_preds = final_model.predict([X_test, Z_test])
        print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")
        y_preds = final_model.predict([X_test, Z_test], batch_size=BATCH_SIZE)
        print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")

        import pathlib
        import zipfile
        from pathlib import Path

        final_model_fp = f"{Path(__file__).parent}/final_model"
        final_model.save(final_model_fp)

        def check_size(filepath):
            du = subprocess.run(
                f"du -sh {filepath} | awk '{{print $1}}'",
                check=True,
                capture_output=True,
                shell=True,
                text=True,
            ).stdout
            return du

        directory = pathlib.Path(final_model_fp)

        with zipfile.ZipFile(
            "compressed_final_model.zip",
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for file_path in directory.rglob("*"):
                archive.write(file_path, arcname=file_path.relative_to(directory))

        import pdb

        original_model_fp = f"{Path(__file__).parent}/models/plasticc/model-1619624444-0.1.dev765+g7c90cbb.d20210428"
        final_compressed_model_fp = (
            f"{Path(__file__).parent}/compressed_final_model.zip"
        )

        print(f"ORIGINAL MODEL ON DISK: {check_size(original_model_fp)}")
        print(f"CLUSTERED MODEL ON DISK: {check_size(final_model_fp)}")
        print(
            f"FINAL COMPRESSED CLUSTERED MODEL ON DISK: {check_size(final_compressed_model_fp)}"
        )


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
