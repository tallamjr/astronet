import argparse
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

from astronet.t2.constants import astronet_working_directory as asnwd
from astronet.t2.preprocess import one_hot_encode
from astronet.t2.utils import t2_logger, load_wisdm_2010, load_wisdm_2019, load_plasticc


def plot_acc_history(dataset, model_name, event, save=True):
    # TODO: Update docstrings
    plt.figure(figsize=(16, 9))
    plt.plot(event['acc'], label='train')
    plt.plot(event['val_acc'], label='validation')
    plt.xlabel("Epoch")
    # plt.xticks(np.arange(len(event['acc'])))
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(r'Training vs. Validation per Epoch')

    if save:
        fname = f"{asnwd}/astronet/t2/plots/{dataset}/model-acc-{model_name}.pdf"
        plt.savefig(fname, format='pdf')
        plt.clf()
    else:
        print(model_name)
        plt.show()


def plot_loss_history(dataset, model_name, event, save=True):
    # TODO: Update docstrings
    plt.figure(figsize=(16, 9))
    plt.plot(event['loss'], label='train')
    plt.plot(event['val_loss'], label='validation')
    plt.xlabel("Epoch")
    # plt.xticks(np.arange(len(event['acc'])))
    plt.ylabel("Loss")
    plt.legend()
    plt.title(r'Training vs. Validation per Epoch')

    if save:
        fname = f"{asnwd}/astronet/t2/plots/{dataset}/model-loss-{model_name}.pdf"
        plt.savefig(fname, format='pdf')
        plt.clf()
    else:
        print(model_name)
        plt.show()


def plot_confusion_matrix(dataset, model_name, y_true, y_pred, class_names, cmap=None, save=True):
    # TODO: Update docstrings
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 10))
    ax = sns.heatmap(
        cm / np.sum(cm, axis=1, keepdims=1),
        annot=True,
        # fmt="d",
        fmt=".2f",
        # cmap=sns.diverging_palette(220, 20, n=7),
        cmap=cmap,
        ax=ax,
    )

    import matplotlib.transforms
    # plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45)

    # Create offset transform by 5 points in y direction
    dy = 20 / 72.0
    dx = 0 / 72.0
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.yaxis.get_majorticklabels(), ha="right")
    # apply offset transform to all x ticklabels.
    for label in ax.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    if save:
        fname = f"{asnwd}/astronet/t2/plots/{dataset}/model-cm-{model_name}.pdf"
        plt.savefig(fname, format='pdf')
        plt.clf()
    else:
        print(model_name)
        plt.show()


def plot_multiROC(dataset, model_name, model, X_test, y_test, class_names, save=True):
    # TODO: Update docstrings
    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = model.predict(X_test)
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
    plt.figure(figsize=(16, 9))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=3)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=3)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC: {0} (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    if save:
        fname = f"{asnwd}/astronet/t2/plots/{dataset}/model-roc-{model_name}.pdf"
        plt.savefig(fname, format='pdf')
        plt.clf()
    else:
        print(model_name)
        plt.show()


if __name__ == '__main__':

    try:
        log = t2_logger(__file__)
        log.info("=" * shutil.get_terminal_size((80, 20))[0])
        log.info(f"File Path: {Path(__file__).absolute()}")
        log.info(f"Parent of Directory Path: {Path().absolute().parent}")
    except:
        print("Seems you are running from a notebook...")
        __file__ = f"{Path().resolve().parent}/astronet/t2/visuals.py"

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.serif": ["Computer Modern Roman"]})

    mpl.style.use("seaborn")

    parser = argparse.ArgumentParser(description='Process named model')

    parser.add_argument('-m', '--model',
            help='Name of tensorflow.keras model, i.e. model-<timestamp>-<hash>')

    parser.add_argument("-d", "--dataset", default="wisdm_2010",
            help="Choose which dataset to use; options include: 'wisdm_2010', 'wisdm_2019'")

    try:
        args = parser.parse_args()
        argsdict = vars(args)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    dataset = args.dataset

    if args.dataset == "wisdm_2010":
        load_dataset = load_wisdm_2010
    elif args.dataset == "wisdm_2019":
        load_dataset = load_wisdm_2019
    elif args.dataset == "plasticc":
        load_dataset = load_plasticc
        class_mapping = {
            15: "TDE",
            42: "SNII",
            52: "SNIax",
            62: "SNIbc",
            64: "KN",
            67: "SNIa-91bg",
            88: "AGN",
            90: "SNIa",
            95: "SLSN-I",
            1: "SNIa",
            2: "SNII",
            3: "SNIbc",
        }

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    # One hot encode y
    enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)
    class_encoding = enc.categories_[0]

    if args.dataset == "plasticc":
        class_names = list(np.vectorize(class_mapping.get)(class_encoding))
    else:
        class_names = class_encoding

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    dataset = args.dataset
    with open(f"{asnwd}/astronet/t2/models/{dataset}/results.json") as f:
        events = json.load(f)
        if args.model:
            # Get params for model chosen with cli args
            event = next(item for item in events['training_result'] if item["name"] == args.model)
            print(event)
        else:
            # Get params for best model with highest test accuracy
            event = max(events['training_result'], key=lambda ev: ev['value'])
            print(event)

    model_name = event['name']

    plot_acc_history(dataset, model_name, event)
    plot_loss_history(dataset, model_name, event)

    model = keras.models.load_model(f"{asnwd}/astronet/t2/models/{dataset}/model-{model_name}")
    y_pred = model.predict(X_test)

    plot_confusion_matrix(
        dataset,
        model_name,
        enc.inverse_transform(y_test),
        enc.inverse_transform(y_pred),
        class_names,  # enc.categories_[0]
    )

    plot_multiROC(dataset, model_name, model, X_test, y_test, class_names)
