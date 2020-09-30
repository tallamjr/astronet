from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

import numpy as np
import json
import tensorflow as tf

from pathlib import Path
print("File      Path:", Path(__file__).absolute())
print("Parent of Directory Path:", Path().absolute().parent)

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

with open(str(Path().absolute()) + '/models/results.json') as f:
    events = json.load(f)
    event = max(events['training_result'], key=lambda ev: ev['value'])
    print(event)

# TODO
# 1. Load saved model.
# 2. Run inference.
# 3. Save plots of performance.
