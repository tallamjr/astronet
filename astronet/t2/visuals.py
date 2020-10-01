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


plot_cm(
  enc.inverse_transform(y_test),
  enc.inverse_transform(y_pred),
  enc.categories_[0]
)
