import argparse
import json
import numpy as np
import shutil
import sys
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from astronet.t2.constants import (
    plasticc_weights_dict,
    astronet_working_directory as asnwd,
)
from astronet.t2.metrics import plasticc_log_loss, custom_log_loss
from astronet.t2.preprocess import one_hot_encode, tf_one_hot_encode
from astronet.t2.utils import (
    t2_logger,
    load_wisdm_2010,
    load_wisdm_2019,
    load_plasticc,
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

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})

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
enc, sk_y_train, sk_y_val, sk_y_test = one_hot_encode(y_train, y_val, y_test)

# One hot encode y with tensorflow
tf_y_train, tf_y_val, tf_y_test = tf_one_hot_encode(y_train, y_val, y_test)
print(y_test)
print(sk_y_train)
print(tf_y_train)

# flat_labels = tf.reshape(y_train, [-1])
# print(flat_labels)
# tfenc = tf.one_hot(flat_labels, len(np.unique(y_train)))
# print(tfenc)

dct = {42: 0, 62: 1, 90: 2}
lst = y_test.flatten().tolist()

flabels = list(map(dct.get, lst))
__y_test = tf.one_hot(flabels, len(np.unique(y_test)))
print(f"FLABELS: {flabels}")

inv_map = {v: k for k, v in dct.items()}
print(inv_map)
print(__y_test.numpy().tolist())

flip = list(map(inv_map.get, flabels))
print(f"FLIP: {flip}")

dataset = args.dataset
with open(f"{asnwd}/astronet/t2/models/{dataset}/results.json") as f:
    events = json.load(f)
    if args.model:
        # Get params for model chosen with cli args
        event = next(item for item in events['training_result'] if item["name"] == args.model)
        # print(event)
    else:
        # Get params for best model with highest test accuracy
        event = min(events['training_result'], key=lambda ev: ev['model_evaluate_on_test_loss'])
        print(event)

model_name = event['name']

model = keras.models.load_model(f"{asnwd}/astronet/t2/models/{args.dataset}/model-{model_name}")

model.evaluate(X_test, sk_y_test)
model.evaluate(X_test, tf_y_test)
y_probs = model.predict(X_test)

cm = confusion_matrix(enc.inverse_transform(sk_y_test), enc.inverse_transform(y_probs))
print(cm / np.sum(cm, axis=1, keepdims=1))

print("             Results for Test Set\n\n" +
      classification_report(enc.inverse_transform(y_probs),
                            enc.inverse_transform(sk_y_test)))


closs = custom_log_loss(tf_y_test, y_probs)
print(f"CUSTOM LOG LOSS {closs}")

# tfloss = tf.keras.losses.categorical_crossentropy(y_test, y_probs)
# print(np.average(tfloss.numpy()))

# cce = tf.keras.losses.CategoricalCrossentropy()
# print(cce(y_test, y_probs).numpy())
