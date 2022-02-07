import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import random as python_random
import sys
import seaborn as sns
import tensorflow as tf

from tensorflow import keras

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.utils import get_encoding, find_optimal_batch_size

from astronet.metrics import WeightedLogLoss
from astronet.viz.visualise_results import (
    plot_acc_history,
    plot_confusion_matrix,
    plot_loss_history,
    plot_multiROC,
    plot_multiPR,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.serif": ["Computer Modern Roman"],
    }
)


class Plots(object):
    # TODO: Update docstrings
    def __init__(self, architecture, dataset, model_name, redshift, savefigs=True):
        self.architecture = architecture
        self.dataset = dataset
        self.model_name = model_name
        self.redshift = redshift
        self.redshift = redshift
        self.savefigs = savefigs

    def __call__(self):
        # architecture = "t2"
        # dataset = "plasticc"
        # X_train, y_train, X_test, y_test, loss, Z_train, Z_test = load_dataset(
        #                                                                         dataset,
        #                                                                         redshift=True,
        #                                                                         avocado=None,
        #                                                                         testset=True
        #                                                         )
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
            inputs = [X_test, Z_test]
            results_filename = f"{asnwd}/astronet/{self.architecture}/models/{dataset}/results_with_z.json"
        else:
            inputs = X_test
            results_filename = (
                f"{asnwd}/astronet/{self.architecture}/models/{dataset}/results.json"
            )

        with open(results_filename) as f:
            events = json.load(f)
            if self.model_name is not None:
                # Get params for model chosen with cli args
                event = next(
                    item
                    for item in events["training_result"]
                    if item["name"] == model_name
                )
            else:
                event = min(
                    events["training_result"],
                    key=lambda ev: ev["model_evaluate_on_test_loss"],
                )

        model = keras.models.load_model(
            f"{asnwd}/astronet/{self.architecture}/models/{dataset}/model-{self.model_name}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

        dataform = "testset"
        encoding, class_encoding, class_names = get_encoding(dataset, dataform=dataform)
        from collections import Counter
        from pandas.core.common import flatten

        # y_true = encoding.inverse_transform(y_train)
        # y_true = encoding.inverse_transform(y_train)
        # print("N_TRAIN:", Counter(list(flatten(y_true))))

        y_true_test = encoding.inverse_transform(y_test)
        print("N_TEST:", Counter(list(flatten(y_true_test))))

        logloss = event["model_evaluate_on_test_loss"]
        acc = event["model_evaluate_on_test_acc"]
        print(f"LogLoss on Test Set: {logloss}, Accuracy on Test Set: {acc}")

        print("Running predictions")
        wloss = WeightedLogLoss()
        y_preds = model.predict([X_test, Z_test])
        print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")
        y_preds = model.predict([X_test, Z_test], batch_size=BATCH_SIZE)
        print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")
        # Causes OOM with all samples -->
        # y_preds = model.predict([X_test, Z_test], batch_size=num_samples)
        # print(f"LL-Test: {wloss(y_test, y_preds).numpy():.2f}")

        # Train predictions
        # y_preds_train = model.predict([X_train, Z_train])
        # print(f"LL-Train: {wloss(y_train, y_preds_train).numpy():.2f}")

        print("Plotting figures...")
        cmap = sns.light_palette("Navy", as_cmap=True)
        plot_confusion_matrix(
            self.architecture,
            dataset,
            model_name,
            y_test,
            y_preds,
            encoding,
            class_names,  # enc.categories_[0]
            save=self.savefigs,
            cmap=cmap,
        )

        plot_acc_history(
            self.architecture, dataset, model_name, event, save=self.savefigs
        )
        plot_loss_history(
            self.architecture, dataset, model_name, event, save=self.savefigs
        )

        plot_multiROC(
            self.architecture,
            dataset,
            model_name,
            model,
            inputs,
            y_test,
            class_names,
            save=self.savefigs,
        )
        plot_multiPR(
            self.architecture,
            dataset,
            model_name,
            model,
            inputs,
            y_test,
            class_names,
            save=self.savefigs,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate trained model for given architecture"
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

    plotting = Plots(
        architecture=architecture,
        dataset=dataset,
        model_name=model_name,
        redshift=redshift,
    )
    plotting()
