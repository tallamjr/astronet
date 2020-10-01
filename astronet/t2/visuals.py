import json
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sn
from pathlib import Path
print("File      Path:", Path(__file__).absolute())
print("Parent of Directory Path:", Path().absolute().parent)
from astronet.t2.utils import train_val_test_split, create_dataset
from astronet.t2.preprocess import robust_scale, one_hot_encode


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Computer Modern Roman"]})
## for Palatino and other serif fonts use:
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })

# TODO:
# 1. Set up arg parse such that plots can be made by passing in a model name, or the best performing model is chosen
# 2. Plot confusion matrix
# 3. Plot history metrics
with open(str(Path().absolute()) + '/models/results.json') as f:
    events = json.load(f)
    event = max(events['training_result'], key=lambda ev: ev['value'])
    print(event)

model_name = event['name']

mpl.style.use("seaborn")

plt.plot(event['acc'], label='train')
plt.plot(event['val_acc'], label='validation')
plt.ylabel("Accuracy")
plt.legend();
plt.title(r'Training vs. Validation per Epoch $\mathbf{W_y(\tau, j=3)}$')

fname = str(Path().absolute()) + f"/plots/model-acc-{model_name}.pdf"
plt.savefig(fname, format='pdf')
plt.clf()
########

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# Load WISDM-2010 or WISDM-2019 dataset
column_names = [
    "user_id",
    "activity",
    "timestamp",
    "x_axis",
    "y_axis",
    "z_axis",
]

df = pd.read_csv(str(Path(__file__).absolute().parent.parent.parent) +
    "/data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",
    header=None,
    names=column_names,
)
df.z_axis.replace(regex=True, inplace=True, to_replace=r";", value=r"")
df["z_axis"] = df.z_axis.astype(np.float64)
df.dropna(axis=0, how="any", inplace=True)

# print(df.head())

cols = ["x_axis", "y_axis", "z_axis"]

# print(df[cols].head())

df_train, df_val, df_test, num_features = train_val_test_split(df, cols)
# print(num_features)  # Should = 3 in this case

# Perfrom robust scaling
robust_scale(df_train, df_val, df_test, cols)

TIME_STEPS = 200
STEP = 40

X_train, y_train = create_dataset(
    df_train[cols],
    df_train.activity,
    TIME_STEPS,
    STEP
)

X_val, y_val = create_dataset(
    df_val[cols],
    df_val.activity,
    TIME_STEPS,
    STEP
)

X_test, y_test = create_dataset(
    df_test[cols],
    df_test.activity,
    TIME_STEPS,
    STEP
)

# print(X_train.shape, y_train.shape)

# One hot encode y
enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)

# print(X_train.shape, y_train.shape)
# print(X_val.shape, y_val.shape)
# print(X_test.shape, y_test.shape)

model = keras.models.load_model(str(Path().absolute()) + f"/models/model-{model_name}")

# model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
cm = confusion_matrix(enc.inverse_transform(y_test), enc.inverse_transform(y_pred))

def plot_cm(y_true, y_pred, class_names):
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(18, 10))
    ax = sns.heatmap(
          cm / np.sum(cm, axis=1, keepdims=1),
          annot=True,
          # fmt="d",
          fmt=".2f",
          # cmap=sns.diverging_palette(220, 20, n=7),
          # cmap="coolwarm",
          ax=ax
          )

    import matplotlib.transforms

    # plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45)

    # Create offset transform by 5 points in y direction
    dy = 20/72.; dx = 0/72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp( ax.yaxis.get_majorticklabels(), ha="right" )
    # apply offset transform to all x ticklabels.
    for label in ax.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    # plt.setp( ax.yaxis.get_majorticklabels(), ha="center" )
    # b, t = plt.ylim() # discover the values for bottom and top
    # b += 0.5 # Add 0.5 to the bottom
    # t -= 0.5 # Subtract 0.5 from the top
    # plt.ylim(b, t) # update the ylim(bottom, top) values
    fname = str(Path().absolute()) + f"/plots/model-cm-{model_name}.pdf"
    plt.savefig(fname, format='pdf')
    plt.clf()

plot_cm(
  enc.inverse_transform(y_test),
  enc.inverse_transform(y_pred),
  enc.categories_[0]
)

# import sklearn.metrics as metrics
# # calculate the fpr and tpr for all thresholds of the classification
# probs = model.predict(X_test)
# print(probs)
# preds = probs[:,1]
# fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
# roc_auc = metrics.auc(fpr, tpr)

# # method I: plt
# import matplotlib.pyplot as plt
# plt.title(r'Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel(r'True Positive Rate')
# plt.xlabel('False Positive Rate')
# fname = str(Path().absolute()) + f"/plots/model-roc-{model_name}.pdf"
# plt.savefig(fname, format='pdf')

import numpy as np
from numpy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
y_score = model.predict(X_test)
y_classes = y_score.argmax(axis=-1)
print(y_classes)
# n_classes = len(np.unique(y_classes))
n_classes = len(enc.categories_[0])
print(enc.categories_[0][0])
print(type(enc.categories_[0]))

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
             ''.format(enc.categories_[0][i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class Receiver Operating Characteristic')
plt.legend(loc="lower right")

fname = str(Path().absolute()) + f"/plots/model-roc-{model_name}.pdf"
plt.savefig(fname, format='pdf')
plt.clf()
