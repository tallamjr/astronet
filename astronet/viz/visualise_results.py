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
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from numpy import interp
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from tensorflow import keras

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.metrics import WeightedLogLoss
from astronet.utils import (
    astronet_logger,
    get_encoding,
    load_plasticc,
    load_wisdm_2010,
    load_wisdm_2019,
)


def plot_acc_history(architecture, dataset, model_name, event, save=True, ax=None):
    # TODO: Update docstrings

    if ax is not None:
        ax = ax or plt.gca()
        ax.plot(event["acc"], label="train")
        ax.plot(event["val_acc"], label="validation")
        ax.set_title(rf"{dataset}")

    else:
        plt.figure(figsize=(16, 9))
        plt.plot(event["acc"], label="train")
        plt.plot(event["val_acc"], label="validation")
        plt.xlabel("Epoch")
        # plt.xticks(np.arange(len(event['acc'])))
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title(rf"Training vs. Validation per Epoch - {dataset}")

    fig = plt.gcf()
    if save:
        try:
            os.makedirs(
                f"{asnwd}/astronet/{architecture}/plots/{dataset}/{model_name}",
                exist_ok=True,
            )
            fname = f"{asnwd}/astronet/{architecture}/plots/{dataset}/{model_name}/model-acc-{model_name}.pdf"
        except Exception:
            fname = f"{asnwd}/astronet/{architecture}/plots/{dataset}/model-acc-{model_name}.pdf"
        plt.savefig(fname, format="pdf")
        plt.clf()

    return fig


def plot_loss_history(architecture, dataset, model_name, event, save=True, ax=None):
    # TODO: Update docstrings

    if ax is not None:
        ax = ax or plt.gca()
        ax.plot(event["loss"], label="train")
        ax.plot(event["val_loss"], label="validation")
        ax.set_title(rf"{dataset}")
    else:
        plt.figure(figsize=(16, 9))
        plt.plot(event["loss"], label="train")
        plt.plot(event["val_loss"], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(r"Training vs. Validation per Epoch")

    fig = plt.gcf()
    if save:
        try:
            os.makedirs(
                f"{asnwd}/astronet/{architecture}/plots/{dataset}/{model_name}",
                exist_ok=True,
            )
            fname = f"{asnwd}/astronet/{architecture}/plots/{dataset}/{model_name}/model-loss-{model_name}.pdf"
        except Exception:
            fname = f"{asnwd}/astronet/{architecture}/plots/{dataset}/model-loss-{model_name}.pdf"
        plt.savefig(fname, format="pdf")
        plt.clf()

    return fig


def plot_confusion_matrix(
    architecture,
    dataset,
    model_name,
    y_test,
    y_preds,
    encoding,
    class_names,
    cmap=None,
    save=True,
):
    # TODO: Update docstrings

    y_true = encoding.inverse_transform(y_test)
    y_pred = encoding.inverse_transform(y_preds)

    sns.set(style="whitegrid", palette="muted", font_scale=1.5)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax = sns.heatmap(
        cm / np.sum(cm, axis=1, keepdims=1),
        annot=True,
        # fmt="d",
        fmt=".2f",
        # cmap=sns.diverging_palette(220, 20, n=7),
        cmap=cmap,
        cbar=True,
        ax=ax,
    )

    # plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45)

    # import matplotlib.transforms
    # Create offset transform by 5 points in y direction
    # dy = 20 / 72.0
    # dx = 0 / 72.0
    # offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    if dataset == "plasticc":
        wloss = WeightedLogLoss()
        wloss = wloss(y_test, y_preds).numpy()
        plt.title(f"Test Set Confusion Matrix, Log Loss = {wloss:.3f}")
    else:
        plt.title(f"Test Set Confusion Matrix -- {dataset}")

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(
        ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor"
    )
    plt.setp(
        ax.yaxis.get_majorticklabels(),
        rotation="horizontal",
        ha="right",
        rotation_mode="anchor",
    )
    # apply offset transform to all x ticklabels.
    # for label in ax.yaxis.get_majorticklabels():
    #     label.set_transform(label.get_transform() + offset)
    plt.tight_layout()
    if save:
        try:
            os.makedirs(
                f"{asnwd}/astronet/{architecture}/plots/{dataset}/{model_name}",
                exist_ok=True,
            )
            fname = f"{asnwd}/astronet/{architecture}/plots/{dataset}/{model_name}/model-cm-{model_name}.pdf"
        except Exception:
            fname = f"{asnwd}/astronet/{architecture}/plots/{dataset}/model-cm-{model_name}.pdf"
        plt.savefig(fname, format="pdf")
        plt.clf()
    else:
        print(model_name)
        plt.show()

    return fig


def plot_confusion_matrix_against_baseline(
    architecture,
    dataset,
    model_name,
    test_ds,
    y_test,
    y_preds,
    encoding,
    class_names,
    cmap=None,
    save=True,
):
    # TODO: Update docstrings

    np.set_printoptions(formatter={"all": lambda x: "{:+}".format(x)})
    # https://stackoverflow.com/a/21132866/4521950
    baseline = f"{asnwd}/astronet/{architecture}/models/{dataset}/tinho/clustered_stripped_fink_model"

    baseline_model = keras.models.load_model(
        baseline,
        custom_objects={"WeightedLogLoss": WeightedLogLoss()},
        compile=False,
    )

    y_true = encoding.inverse_transform(y_test)
    y_pred = encoding.inverse_transform(y_preds)

    baseline_y_preds = baseline_model.predict(test_ds, verbose=2)
    baseline_y_pred = encoding.inverse_transform(baseline_y_preds)

    sns.set(style="whitegrid", palette="muted", font_scale=1.5)

    cm = confusion_matrix(y_true, y_pred)
    cm = cm / np.sum(cm, axis=1, keepdims=1)

    cmb = confusion_matrix(y_true, baseline_y_pred)
    cmb = cmb / np.sum(cmb, axis=1, keepdims=1)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax = sns.heatmap(
        np.subtract(cmb, cm),
        annot=True,
        # fmt="d",
        fmt=".2f",
        # cmap=sns.diverging_palette(220, 20, n=7),
        cmap=cmap,
        cbar=True,
        ax=ax,
    )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    if dataset == "plasticc":
        wloss = WeightedLogLoss()
        wloss = wloss(y_test, y_preds).numpy()
        plt.title(f"Test Set Confusion Matrix, Log Loss = {wloss:.3f}")
    else:
        plt.title(f"Test Set Confusion Matrix -- {dataset}")

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(
        ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor"
    )
    plt.setp(
        ax.yaxis.get_majorticklabels(),
        rotation="horizontal",
        ha="right",
        rotation_mode="anchor",
    )
    plt.tight_layout()
    if save:
        try:
            os.makedirs(
                f"{asnwd}/astronet/{architecture}/plots/{dataset}/{model_name}",
                exist_ok=True,
            )
            fname = f"{asnwd}/astronet/{architecture}/plots/{dataset}/{model_name}/model-cm-{model_name}-CMB.pdf"
        except Exception:
            fname = f"{asnwd}/astronet/{architecture}/plots/{dataset}/model-cm-{model_name}-CMB.pdf"
        plt.savefig(fname, format="pdf")
        plt.clf()
    else:
        print(model_name)
        plt.show()

    return fig


def plot_multiROC(
    architecture,
    dataset,
    model_name,
    model,
    y_test,
    y_score,
    class_names,
    save=True,
    colors=plt.cm.Accent.colors,
):
    # TODO: Update docstrings
    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # y_score = model.predict(X_test)
    n_classes = len(class_names)
    # print(enc.categories_[0][0])
    # print(type(enc.categories_[0]))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(12, 9))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-Average ROC curve (area = {0:0.2f})" "".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=3,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-Average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=3,
    )

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors)
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label="ROC: {0} (area = {1:0.2f})" "".format(class_names[i], roc_auc[i]),
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    fig = plt.gcf()

    if save:
        try:
            os.makedirs(
                f"{asnwd}/astronet/{architecture}/plots/{dataset}/{model_name}",
                exist_ok=True,
            )
            fname = f"{asnwd}/astronet/{architecture}/plots/{dataset}/{model_name}/model-roc-{model_name}.pdf"
        except Exception:
            fname = f"{asnwd}/astronet/{architecture}/plots/{dataset}/model-roc-{model_name}.pdf"
        plt.savefig(fname, format="pdf")
        plt.clf()
    else:
        print(model_name)
        plt.show()

    return fig


def plot_multiPR(
    architecture,
    dataset,
    model_name,
    model,
    y_test,
    y_score,
    class_names,
    save=True,
    colors=plt.cm.tab20.colors,
):
    # TODO: Update docstrings
    # Plot linewidth.
    lw = 2
    plt.figure(figsize=(12, 12))

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    # y_score = model.predict(X_test)
    n_classes = len(class_names)

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_test, y_score, average="micro"
    )
    lines = []
    labels = []
    (l,) = plt.plot(
        recall["micro"], precision["micro"], color="deeppink", linestyle=":", lw=lw
    )
    lines.append(l)
    labels.append(
        "micro-Average Precision-Recall (area = {0:0.2f})"
        "".format(average_precision["micro"])
    )

    for i, color in zip(range(n_classes), colors):
        (l,) = plt.plot(recall[i], precision[i], color=color, lw=lw)
        lines.append(l)
        labels.append(
            "Precision-Recall for {0} (area = {1:0.2f})"
            "".format(class_names[i], average_precision[i])
        )

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multi-Class Precision vs. Recall")
    plt.legend(
        lines,
        labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.40),
        fancybox=False,
        shadow=False,
        # prop=dict(size=14),
    )

    if save:
        try:
            os.makedirs(
                f"{asnwd}/astronet/{architecture}/plots/{dataset}/{model_name}",
                exist_ok=True,
            )
            fname = f"{asnwd}/astronet/{architecture}/plots/{dataset}/{model_name}/model-pr-{model_name}.pdf"
        except Exception:
            fname = f"{asnwd}/astronet/{architecture}/plots/{dataset}/model-pr-{model_name}.pdf"
        plt.savefig(fname, format="pdf", bbox_inches="tight")
        plt.clf()
    else:
        print(model_name)
        plt.show()

    return fig


if __name__ == "__main__":

    try:
        log = astronet_logger(__file__)
        log.info("=" * shutil.get_terminal_size((80, 20))[0])
        log.info(f"File Path: {Path(__file__).absolute()}")
        log.info(f"Parent of Directory Path: {Path().absolute().parent}")
    except Exception as e:
        print(f"{e}: Seems you are running from a notebook...")
        __file__ = f"{Path().resolve().paren.parentt}/astronet/visualise_results.py"

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )

    mpl.style.use("seaborn")

    parser = argparse.ArgumentParser(description="Process named model")

    parser.add_argument("-a", "--architecture", help="Choice of atx or t2 architecture")

    parser.add_argument(
        "-m",
        "--model",
        help="Name of tensorflow.keras model, i.e. model-<timestamp>-<hash>",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        default="wisdm_2010",
        help="Choose which dataset to use; options include: 'wisdm_2010', 'wisdm_2019'",
    )

    try:
        args = parser.parse_args()
        argsdict = vars(args)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    architecture = args.architecture
    dataset = args.dataset

    if args.dataset == "wisdm_2010":
        load_dataset = load_wisdm_2010
    elif args.dataset == "wisdm_2019":
        load_dataset = load_wisdm_2019
    elif args.dataset == "plasticc":
        load_dataset = load_plasticc
        dataform = "full"

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Get inverse of one-hot encoding
    enc, class_encoding, class_names = get_encoding(dataset, dataform=dataform)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    dataset = args.dataset
    with open(f"{asnwd}/astronet/{architecture}/models/{dataset}/results.json") as f:
        events = json.load(f)
        if args.model:
            # Get params for model chosen with cli args
            event = next(
                item for item in events["training_result"] if item["name"] == args.model
            )
            print(event)
        else:
            # Get params for best model with highest test accuracy
            event = max(events["training_result"], key=lambda ev: ev["value"])
            print(event)

    model_name = event["name"]

    plot_acc_history(dataset, model_name, event)
    plot_loss_history(dataset, model_name, event)

    model = keras.models.load_model(
        f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}"
    )
    y_pred = model.predict(X_test)

    plot_confusion_matrix(
        dataset,
        model_name,
        enc.inverse_transform(y_test),
        enc.inverse_transform(y_pred),
        class_names,  # enc.categories_[0]
    )

    plot_multiROC(dataset, model_name, model, X_test, y_test, class_names)
