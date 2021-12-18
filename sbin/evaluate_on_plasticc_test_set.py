import argparse
import joblib
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shutil
import sys
import seaborn as sns
import tensorflow as tf

from itertools import cycle
from numpy import interp
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from tensorflow import keras

from astronet.constants import astronet_working_directory as asnwd
from astronet.preprocess import one_hot_encode
from astronet.utils import astronet_logger, load_dataset, find_optimal_batch_size

from astronet.metrics import WeightedLogLoss
from astronet.visualise_results import (
    plot_acc_history,
    plot_confusion_matrix,
    plot_loss_history,
    plot_multiROC,
    plot_multiPR,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

import random as python_random
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Computer Modern Roman"]})

print(plt.style.available)


class Plots(object):
    # TODO: Update docstrings
    def __init__(self, architecture, dataset, model_name, redshift):
        self.architecture = architecture
        self.dataset = dataset
        self.model_name = model_name
        self.redshift = redshift

    def __call__(self):
# architecture = "t2"
# dataset = "plasticc"
# X_train, y_train, X_test, y_test, loss, Z_train, Z_test = load_dataset(
#                                                                         dataset,
#                                                                         redshift=True,
#                                                                         avocado=None,
#                                                                         testset=True
#                                                         )
# print(f"""
#         X_TRAIN: {X_train.shape}, Y_TRAIN: {y_train.shape},\n
#         X_TEST: {X_test.shape}, Y_TEST: {y_test.shape},\n
#         Z_TRAIN: {Z_train.shape}, Z_TEST: {Z_test.shape}\n
#         """
# )

        X_test = np.load(
                f"{asnwd}/data/plasticc/test_set/infer/X_test.npy",
        )
        y_test = np.load(
                f"{asnwd}/data/plasticc/test_set/infer/y_test.npy",
        )
        Z_test = np.load(
                f"{asnwd}/data/plasticc/test_set/infer/Z_test.npy",
        )

        # num_classes = y_train.shape[1]
        num_classes = y_test.shape[1]
        num_samples, timesteps, num_features = X_test.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)
        BATCH_SIZE = find_optimal_batch_size(num_samples)
        print(f"BATCH_SIZE:{BATCH_SIZE}")

        if self.redshift is not None:
            inputs = [X_test, Z_test]
        # model_name = "1614711597-1ba461b"
        # model_name = "1615402794-0.1.dev686+g8ce4a41"
        # model_name = "1615495431-0.1.dev701+g2539a82.d20210311" # <-- 0.505
        # model_name = "1619538848-0.1.dev753+g8c73954"
        # model_name = "1613551066-32f3933"
        # model_name = None
        # model_name = "1619624444-0.1.dev765+g7c90cbb.d20210428"
            results_filename = f"{asnwd}/astronet/{architecture}/models/{dataset}/results_with_z.json"
        else:
            inputs = X_test
            results_filename = f"{asnwd}/astronet/{architecture}/models/{dataset}/results_with_z.json"
            #     else:
            #         # Get params for best model with lowest loss
            #         event = min(
            #             (item for item in events["training_result"] if item["augmented"] is None),
            #                 key=lambda ev: ev["model_evaluate_on_test_loss"],
            #             )

            #         event = min(events['training_result'], key=lambda ev: ev['model_evaluate_on_test_loss'])
        with open(results_filename) as f:
            events = json.load(f)
            if self.model_name is not None:
                # Get params for model chosen with cli args
                event = next(item for item in events['training_result'] if item["name"] == model_name)
            else:
                event = min(events['training_result'], key=lambda ev: ev['model_evaluate_on_test_loss'])

        model = keras.models.load_model(
            f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{self.model}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

        dataform = "testset"
        with open(f"{asnwd}/data/{dataform}-{dataset}.encoding", "rb") as eb:
            encoding = joblib.load(eb)
        class_encoding = encoding.categories_[0]

        if dataset == "plasticc":
            class_mapping = {
                90: "SNIa",
                67: "SNIa-91bg",
                52: "SNIax",
                42: "SNII",
                62: "SNIbc",
                95: "SLSN-I",
                15: "TDE",
                64: "KN",
                88: "AGN",
                92: "RRL",
                65: "M-dwarf",
                16: "EB",
                53: "Mira",
                6: "$\mu$-Lens-Single",
            }
            class_encoding
            class_names = list(np.vectorize(class_mapping.get)(class_encoding))
        else:
            class_names = class_encoding
        # from sklearn.preprocessing import OneHotEncoder
        # enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        # enc = enc.fit(y_test)
        # # y_full_test_true_no_99 = enc.transform(y_full_test_no_99)
        # class_encoding = enc.categories_[0]
        # if dataset == "plasticc":
        #     class_mapping = {
        #         90: "SNIa",
        #         67: "SNIa-91bg",
        #         52: "SNIax",
        #         42: "SNII",
        #         62: "SNIbc",
        #         95: "SLSN-I",
        #         15: "TDE",
        #         64: "KN",
        #         88: "AGN",
        #         92: "RRL",
        #         65: "M-dwarf",
        #         16: "EB",
        #         53: "Mira",
        #         6: "$\mu$-Lens-Single",
        #     }
        #     class_encoding
        #     class_names = list(np.vectorize(class_mapping.get)(class_encoding))
        # else:
        #     class_names = class_encoding

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
            architecture,
            dataset,
            model_name,
            y_test,
            y_preds,
            encoding,
            class_names,  # enc.categories_[0]
            save=True,
            cmap=cmap
        )

        plot_acc_history(architecture, dataset, model_name, event, save=True)
        plot_loss_history(architecture, dataset, model_name, event, save=True)

        plot_multiROC(architecture, dataset, model_name, model, inputs, y_test, class_names, save=True)
        plot_multiPR(architecture, dataset, model_name, model, inputs, y_test, class_names, save=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate trained model for given architecture')

    parser.add_argument("-a", "--architecture", default="t2",
            help="Choose which architecture to evaluate : 'atx', 't2'")

    parser.add_argument("-d", "--dataset", default="plasticc",
            help="Choose which dataset to use: This is fixed for plasticc for now")

    parser.add_argument('-m', '--model', default=None,
            help='Name of tensorflow.keras model, i.e. model-<timestamp>-<hash>')

    parser.add_argument("-z", "--redshift", default=None,
            help="Whether to include redshift features or not")

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
        architecture=architecture, dataset=dataset, model_name=model_name, redshift=redshift
    )
    plotting()
# from pathlib import Path
# for filename in Path(f"{asnwd}/logs").rglob(f"training-*-{model_name}.log"):
#     logfile = filename
# event = pd.read_csv(logfile)

# cmap = sns.light_palette("Navy", as_cmap=True)
# plot_confusion_matrix(
#     dataset,
#     model_name,
#     y_full_test_true_no_99,
#     y_preds,
#     enc,
#     class_names,  # enc.categories_[0]
#     save=True,
#     cmap=cmap
# )

# plot_multiROC(dataset, model_name, model, [X_full_test_no_99, Z_full_test_no_99],
#         y_full_test_true_no_99, class_names, save=True)
