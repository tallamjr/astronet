import argparse
import json
import numpy as np
import sys
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from astronet.t2.utils import t2_logger, load_wisdm_2010, load_wisdm_2019
from astronet.t2.preprocess import one_hot_encode

try:
    log = t2_logger(__file__)
    log.info("_________________________________")
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

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# One hot encode y
enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)

dataset = args.dataset
with open(f"{Path(__file__).absolute().parent}/models/{dataset}/results.json") as f:
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

model = keras.models.load_model(f"{Path(__file__).absolute().parent}/models/{args.dataset}/model-{model_name}")

model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
cm = confusion_matrix(enc.inverse_transform(y_test), enc.inverse_transform(y_pred))
print(cm / np.sum(cm, axis=1, keepdims=1))

print("             Results for Test Set\n\n" +
      classification_report(enc.inverse_transform(y_pred),
                            enc.inverse_transform(y_test)))
