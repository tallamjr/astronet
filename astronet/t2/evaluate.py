from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from astronet.t2.utils import t2_logger, load_WISDM
from astronet.t2.preprocess import one_hot_encode
from astronet.t2.preprocess import robust_scale, one_hot_encode

import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras

from pathlib import Path

log = t2_logger(__file__)
log.info("_________________________________")
log.info("File      Path:" + str(Path(__file__).absolute()))
log.info("Parent of Directory Path:" + str(Path().absolute().parent))

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Load WISDM-2010
X_train, y_train, X_val, y_val, X_test, y_test = load_WISDM()
# One hot encode y
enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)

with open(str(Path().absolute()) + '/models/results.json') as f:
    events = json.load(f)
    event = max(events['training_result'], key=lambda ev: ev['value'])
    print(event)

model_name = event['name']

model = keras.models.load_model(str(Path().absolute()) + f"/models/model-{model_name}")

model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
cm = confusion_matrix(enc.inverse_transform(y_test), enc.inverse_transform(y_pred))
print(cm / np.sum(cm, axis=1, keepdims=1))

print("             Results for Test Set\n\n" +
      classification_report(enc.inverse_transform(y_pred),
                            enc.inverse_transform(y_test)))
