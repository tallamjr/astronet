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
from astronet.t2.utils import t2_logger, load_wisdm_2010, load_wisdm_2019, load_plasticc

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
y_pred = model.predict(X_test)

cm = confusion_matrix(enc.inverse_transform(y_test), enc.inverse_transform(y_pred))
print(cm / np.sum(cm, axis=1, keepdims=1))

print("             Results for Test Set\n\n" +
      classification_report(enc.inverse_transform(y_pred),
                            enc.inverse_transform(y_test)))


probs = y_pred
print(f"PROBS --> {probs}")
y_true = enc.inverse_transform(y_test)
print(f"y_true --> {y_true}")

"""Implementation of weighted log loss used for the Kaggle challenge.

Parameters
----------
y_true: np.array of shape (# samples,)
    Array of the true classes
probs : np.array of shape (# samples, # features)
    Class probabilities for each sample. The order of the classes corresponds to
    that in the attribute `classes_` of the classifier used.

Returns
-------
float
    Weighted log loss used for the Kaggle challenge
"""
predictions = probs.copy()
labels = np.unique(y_true) # assumes the probabilities are also ordered in the same way
# labels = np.unique(enc.inverse_transform(y_test))
print(f"LABELS--> {labels}")

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

# sanitize predictions
epsilon = sys.float_info.epsilon  # this is machine dependent but essentially prevents log(0)
predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
print(f"PREDICTIONS--> {predictions}")
print(f"PREDICTIONS.SHAPE--> {predictions.shape}")
predictions = predictions / np.sum(predictions, axis=1)[:, np.newaxis]
print(f"PREDICTIONS.SHAPE--> {predictions.shape}")
predictions = np.log(predictions) # logarithm because we want a log loss
print(f"PREDICTIONS.LOG.SHAPE--> {predictions.shape}")

class_logloss, weights = [], [] # initialize the classes logloss and weights
print(f"PREDICTIONS.[1].SHAPE--> {np.shape(predictions)[1]}")
for i in range(np.shape(predictions)[1]): # run for each class
    current_label = labels[i]
    # import pdb; pdb.set_trace()
    result = np.average(predictions[y_true.ravel()==current_label, i]) # only those events are from that class
    print(f"RESULT.SHAPE--> {result}")
    class_logloss.append(result)
    weights.append(weights_dict[current_label])
print(class_logloss)
print(weights)

print(-1 * np.average(class_logloss, weights=weights))
print("=" * shutil.get_terminal_size((80, 20))[0])

class_weight = {
    42: 1,
    62: 1,
    90: 1,
}
"""
@author olivier https://www.kaggle.com/ogrellier
multi logloss for PLAsTiCC challenge
"""
# class_weights taken from Giba's topic : https://www.kaggle.com/titericz
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
# with Kyle Boone's post https://www.kaggle.com/kyleboone

y_p = probs
# Trasform y_true in dummies
y_ohe = pd.get_dummies(y_true.ravel())
# Normalize rows and limit y_preds to 1e-15, 1-1e-15
y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
# Transform to log
y_p_log = np.log(y_p)
# Get the log for ones, .values is used to drop the index of DataFrames
# Exclude class 99 for now, since there is no class99 in the training set
# we gave a special process for that class
y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
# Get the number of positives for each class
nb_pos = y_ohe.sum(axis=0).values.astype(float)
# Weight average and divide by the number of positives
class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
y_w = y_log_ones * class_arr / nb_pos

loss = - np.sum(y_w) / np.sum(class_arr)
# assert loss == -1 * np.average(class_logloss, weights=weights)
print(loss)
