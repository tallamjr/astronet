import argparse
import json
import numpy as np
import pandas as pd
import shutil
import sys
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from astronet.t2.constants import astronet_working_directory as asnwd
from astronet.t2.preprocess import one_hot_encode
from astronet.t2.utils import (
    t2_logger,
    load_wisdm_2010,
    load_wisdm_2019,
    load_plasticc,
    plasticc_log_loss,
)

try:
    log = t2_logger(__file__)
    log.info("=" * shutil.get_terminal_size((80, 20))[0])
    log.info(f"File Path: {Path(__file__).absolute()}")
    log.info(f"Parent of Directory Path: {Path().absolute().parent}")
except:
    print("Seems you are running from a notebook...")
    __file__ = f"{Path().resolve().parent}/astronet/t2/evaluate.py"

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

parser = argparse.ArgumentParser(description='Evaluate best performing model for a given dataset')

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

if args.dataset == "wisdm_2010":
    load_dataset = load_wisdm_2010
elif args.dataset == "wisdm_2019":
    load_dataset = load_wisdm_2019
elif args.dataset == "plasticc":
    load_dataset = load_plasticc

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# One hot encode y
enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)

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

model = keras.models.load_model(f"{asnwd}/astronet/t2/models/{args.dataset}/model-{model_name}")

model.evaluate(X_test, y_test)
y_probs = model.predict(X_test)

cm = confusion_matrix(enc.inverse_transform(y_test), enc.inverse_transform(y_probs))
print(cm / np.sum(cm, axis=1, keepdims=1))

print("             Results for Test Set\n\n" +
      classification_report(enc.inverse_transform(y_probs),
                            enc.inverse_transform(y_test)))

weights_dict = {
    6: 1 / 18,
    15: 1 / 9,
    16: 1 / 18,
    42: 1 / 18,
    52: 1 / 18,
    53: 1 / 18,
    62: 1 / 18,
    64: 1 / 9,
    65: 1 / 18,
    67: 1 / 18,
    88: 1 / 18,
    90: 1 / 18,
    92: 1 / 18,
    95: 1 / 18,
    99: 1 / 19,
    1: 1 / 18,
    2: 1 / 18,
    3: 1 / 18,
}


def custom_log_loss(y_true, y_pred):
    """
    References:
    -----------
    -

    """

    yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
    loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.math.log(yc), axis=0) * len(y_pred)))
    # labels = np.unique(y_true)
    # loss = tf.reduce_sum(tf.multiply(- labels, tf.math.log(y_pred))) / len(y_pred)
    return loss


print(custom_log_loss(y_test, y_probs))


# def custom_plasticc_loss(y_true, y_pred):

#     predictions = y_pred
#     labels = np.unique(y_true)

#     # sanitize predictions
#     epsilon = (
#         sys.float_info.epsilon
#     )  # this is machine dependent but essentially prevents log(0)
#     predictions = tf.experimental.numpy.clip(predictions, epsilon, 1.0 - epsilon)
#     predictions = predictions / tf.experimental.numpy.sum(predictions, axis=1)[:, tf.newaxis]
#     predictions = tf.math.log(predictions)  # logarithm because we want a log loss

#     class_logloss, weights = [], []  # initialize the classes logloss and weights
#     for i in range(np.shape(predictions)[1]):  # run for each class
#         current_label = labels[i]
#         result = tf.experimental.numpy.average(
#             predictions[y_true.ravel() == current_label, i]
#         )
#         # works like a boolean mask to provide results for current class. ravel() required to fix
#         # IndexError: result = np.average(predictions[y_true==current_label, i]) # only those
#         # events are from that class IndexError: too many indices for array: array is 2-dimensional,
#         # but 3 were indexed

#         class_logloss.append(result)
#         weights.append(weights_dict[current_label])

#     return -1 * tf.experimental.numpy.average(class_logloss, weights=weights)


# print(custom_plasticc_loss(y_test, y_pred))

if args.dataset == "plasticc":

    y_true = enc.inverse_transform(y_test)

    loss = plasticc_log_loss(y_true, y_probs)
    print(loss)

tfloss = tf.keras.losses.categorical_crossentropy(y_test, y_probs)
print(np.average(tfloss.numpy()))

cce = tf.keras.losses.CategoricalCrossentropy()
print(cce(y_test, y_probs).numpy())
