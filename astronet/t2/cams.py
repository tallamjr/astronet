import argparse
import joblib
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
import sys
import seaborn as sns
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from itertools import cycle
from numpy import interp
from matplotlib import rcParams
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from tensorflow import keras

from astronet.constants import astronet_working_directory as asnwd
from astronet.metrics import WeightedLogLoss
from astronet.preprocess import one_hot_encode
from astronet.t2.model import T2Model
from astronet.utils import astronet_logger, load_dataset, find_optimal_batch_size
from astronet.visualise_results import (
    plot_acc_history,
    plot_confusion_matrix,
    plot_loss_history,
    plot_multiROC,
    _get_encoding,
)

tf.get_logger().setLevel('ERROR')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

import random as python_random
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Computer Modern Roman"]})

plt.rcParams["figure.figsize"] = (20,3)
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=28)
mpl.rc('ytick', labelsize=28)


print(plt.style.available)

mpl.style.use("seaborn-whitegrid")

architecture = "t2"

dataset = "plasticc"

snonly = None

if snonly is not None:
    dataform = "snonly"
else:
    dataform = "full"

# X_train, y_train, X_test, y_test, loss, Z_train, Z_test = load_dataset(
#     dataset, redshift=True, snonly=snonly, testset=None,
# )

X_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/X_test.npy",
)
y_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/y_test.npy",
)
Z_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/Z_test.npy",
)

num_classes = y_test.shape[1]
print(num_classes)

BATCH_SIZE = find_optimal_batch_size(X_test.shape[0])
_, timesteps, num_features = X_test.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)
input_shape = (BATCH_SIZE, timesteps, num_features)
print(input_shape)

_, num_z_features = Z_test.shape
Z_input_shape = (BATCH_SIZE, num_z_features)

model_name = "1619624444-0.1.dev765+g7c90cbb.d20210428"

with open(f"{asnwd}/astronet/{architecture}/models/{dataset}/results_with_z.json") as f:
    events = json.load(f)
    if model_name is not None:
    # Get params for model chosen with cli args
        event = next(item for item in events['training_result'] if item["name"] == model_name)
    else:
        # Get params for best model with lowest loss
        event = min(
            (item for item in events["training_result"] if item["augmented"] is None),
                key=lambda ev: ev["model_evaluate_on_test_loss"],
            )

#         event = min(events['training_result'], key=lambda ev: ev['model_evaluate_on_test_loss'])

model_name = event['name']

embed_dim = event['embed_dim']  # --> Embedding size for each token
num_heads = event['num_heads']  # --> Number of attention heads
ff_dim = event['ff_dim']  # --> Hidden layer size in feed forward network inside transformer

# --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
num_filters = embed_dim

num_layers = event['num_layers']    # --> N x repeated transformer blocks
droprate = event['droprate']        # --> Rate of neurons to drop

input_shape_nobatch = input_shape[1:]
Z_input_shape_nobatch = Z_input_shape[1:]

inputs = [
    tf.keras.Input(shape=input_shape_nobatch),
    tf.keras.Input(shape=Z_input_shape_nobatch),
]

# input_shape_nobatch = input_shape[1:]
# inputs = tf.keras.Input(shape=input_shape_nobatch)

print(input_shape_nobatch, Z_input_shape_nobatch)
print(input_shape)

print(inputs)

tf.config.run_functions_eagerly(True)

model = T2Model(
    input_dim=input_shape,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_filters=num_filters,
    num_classes=num_classes,
    num_layers=num_layers,
    droprate=droprate,
    num_aux_feats=2,
    add_aux_feats_to="L",
)

model.call(inputs, training=True)
model.build(input_shape)
print(model.summary())

model.load_weights(f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}")

print(model.layers)

for i in model.layers:
    print(i.output)

encoding, class_encoding, class_names = _get_encoding(dataset, dataform=dataform)
class_mapping = {
    90: "SNIa",
    67: "SNIa-91bg",
    52: "SNIax",
    42: "SNII",
    62: "SNIbc",
    95: "SLSN-I",
    15: "TDE",
    64: "KN",
    88: "AGN",
    92: "RRL",
    65: "M-dwarf",
    16: "EB",
    53: "Mira",
    6: "$\mu$-Lens-Single",
}
class_encoding
class_names = list(np.vectorize(class_mapping.get)(class_encoding))
print(class_names)

for i in range(len(model.layers)):
    print(i, model.layers[i].name)

from keras.models import Model
# same as previous model but with an additional output
cam_model = Model(inputs=inputs,outputs=(model.layers[2].output,model.layers[5].output), name="CAM")
print(cam_model.summary())

# get the features and results of the test images using the newly created model
features,results = cam_model.predict([X_test, Z_test])

# shape of the features
print("features shape: ", features.shape)
print("results shape", results.shape)

# these are the weights going into the softmax layer
last_dense_layer = model.layers[-1]

# get the weights list.  index 0 contains the weights, index 1 contains the biases
gap_weights_l = last_dense_layer.get_weights()

print("gap_weights_l index 0 contains weights ", gap_weights_l[0].shape)
print("gap_weights_l index 1 contains biases ", gap_weights_l[1].shape)

# shows the number of features per class, and the total number of classes
# Store the weights
gap_weights = gap_weights_l[0]

print(f"There are {gap_weights.shape[0]} feature weights and {gap_weights.shape[1]} classes.")

print(features.shape)

# Get the features for the image at index 0
idx = 0
features_for_img = features[idx,:,:]

print(f"The features for image index {idx} has shape (timesteps, num of feature channels) : ", features_for_img.shape)

# Select the weights that are used for a specific class (0...9)
class_id = 0
# take the dot product between the scaled image features and the weights for
gap_weights_for_one_class = gap_weights[:,class_id]

print("features_for_img_scaled has shape ", features_for_img.shape)
print("gap_weights_for_one_class has shape ", gap_weights_for_one_class.shape)
# take the dot product between the scaled features and the weights for one class
cam = np.dot(features_for_img, gap_weights_for_one_class)
print("class activation map shape ", cam.shape)

cam_all = np.dot(features, gap_weights)
print("all class activation map shape ", cam_all.shape)

from scipy.special import softmax
np.set_printoptions(precision=15)
pd.options.display.float_format = '{:.15f}'.format

cam_all_softmax = softmax(cam_all, axis=1)
print(cam_all_softmax.shape)

(num_objects, num_cam_features, num_classes) = cam_all.shape

import numpy.testing as npt
# npt.assert_almost_equal((cam_all.shape[0] * cam_all.shape[2]), cam_all_softmax.sum(), decimal=1)
npt.assert_almost_equal((num_objects * num_classes), cam_all_softmax.sum(axis=1).sum(), decimal=1)

# camr = cam_all_softmax[:,100:102,:]
camr = cam_all_softmax[:,:,:]

# df = pd.DataFrame(data=camr.reshape(27468,2), columns=["redshift", "redshift_error"])
df = pd.DataFrame(data=camr.reshape((num_objects * num_classes), num_cam_features))

data = pd.DataFrame(columns=df.columns)
for i, chunk in enumerate(np.array_split(df, 14)):
#     print(chunk.shape, i+1)
    chunk["class"] = class_names[i]
#     chunk["class"] = i
    assert len(chunk) == len(df) / 14
    data = pd.concat([data, chunk])

assert data.shape == ((num_objects * num_classes), (num_cam_features + 1))

for i, chunk in enumerate(np.array_split(data, 1)):
    print(chunk.shape)
    chunk["class"] = "All Classes"
    data_all = pd.concat([data, chunk])

# data = data.rename(columns={100: "redshift", 101: "redshift-error"})
data_all.rename(
    columns={100: "redshift", 101: "redshift-error"}, inplace=True
)
# data_all = data_all.rename(columns={100: "redshift", 101: "redshift-error"})

df = data_all.filter(
    items=[
        "redshift",
        "redshift-error",
    ]
)

# figure size in inches
rcParams["figure.figsize"] = 16, 9
rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Computer Modern Roman"]})
sns.set_theme(style="whitegrid")

class_names.append("All Classes")
print(class_names)

assert len(class_names) == 15
######################################################################################
ax = sns.violinplot(x=df["class"], y=df["redshift-error"], inner="box", cut=0)

ax.set_title(r'Attention Weight Distriubtion Per Class - Redshift Error', fontsize=16)
ax.set_xlabel('Class', fontsize=16)
ax.set_xticklabels(class_names)
ax.set_ylabel('Attention Weight Percentage', fontsize=16)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.set(ylim=(0, 0.04))
fig = ax.get_figure()
plt.savefig(
    f"{asnwd}/astronet/t2/plots/plasticc/cams/cam-violin-redshift-error-per-class.pdf",
    format="pdf",
)
plt.clf()
######################################################################################
ax = sns.violinplot(x=df["class"], y=df['redshift'], inner="box", cut=0)

ax.set_title(r'Attention Weight Distriubtion Per Class - Redshift', fontsize=16)
ax.set_xlabel('Class', fontsize=16)
ax.set_xticklabels(class_names)
ax.set_ylabel('Attention Weight Percentage', fontsize=16)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
ax.set(ylim=(0, 0.04))
fig = ax.get_figure()
plt.savefig(
    f"{asnwd}/astronet/t2/plots/plasticc/cams/cam-violin-redshift-per-class.pdf",
    format="pdf",
)
plt.clf()
######################################################################################
# ax = sns.violinplot(data=data["redshift"], inner="box", cut=0)

# ax.set_title(r'Attention Weight Distriubtion - Redshift', fontsize=16)
# ax.set_xlabel('All Classes', fontsize=16)
# ax.set_xticks([])
# ax.set_ylabel('Attention Weight Percentage', fontsize=16)
# ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
# fig = ax.get_figure()
# plt.savefig(
#     f"{asnwd}/astronet/t2/plots/plasticc/cams/cam-violin-redshift-all-data.pdf",
#     format="pdf",
# )
# plt.clf()
######################################################################################
# ax = sns.violinplot(data=data["redshift-error"], inner="box", cut=0)

# ax.set_title(r'Attention Weight Distriubtion - Redshift Error', fontsize=16)
# ax.set_xlabel('All Classes', fontsize=16)
# ax.set_xticks([])
# ax.set_ylabel('Attention Weight Percentage', fontsize=16)
# ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
# fig = ax.get_figure()
# plt.savefig(
#     f"{asnwd}/astronet/t2/plots/plasticc/cams/cam-violin-redshift-error-all-data.pdf",
#     format="pdf",
# )
# plt.clf()
######################################################################################

df = data[data["redshift"] < 0.0001]
print(df['redshift'].dtypes, df.shape)

df = data[data["redshift-error"] < 0.0001]
print(df['redshift-error'].dtypes, df.shape)


def show_cam(image_index, desired_class, counter):
    '''displays the class activation map of a particular image'''

  # takes the features of the chosen image
    features_for_img = features[image_index,:,:]

  # get the class with the highest output probability
    prediction = np.argmax(results[image_index])

  # get the gap weights at the predicted class
    class_activation_weights = gap_weights[:,prediction]

  # upsample the features to the image's original size (28 x 28)
#   class_activation_features = sp.ndimage.zoom(features_for_img, (28/3, 28/3, 1), order=2)
    class_activation_features = features_for_img

  # compute the intensity of each feature in the CAM
    cam_output = np.dot(class_activation_features,class_activation_weights)
    print(cam_output.shape)
    print(np.expand_dims(cam_output, axis=0).shape)
    cam_output = np.expand_dims(cam_output, axis=0)
    print('Predicted Class = ' +str(prediction)+ ', Probability = ' + str(results[image_index][prediction]))

    if (results[image_index][prediction] < 0.90):
        return False

    my_cmap = sns.light_palette("Navy", as_cmap=True)
    fig, ax = plt.subplots(figsize=(20, 8))
    from scipy.special import softmax
    sns.heatmap(softmax(cam_output), cmap=my_cmap, cbar=False, robust=True, ax=ax, annot=False, fmt=".1%",) # vmin=v.min(), vmax=v.max()
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0%", "100%"])
    ax2 = ax.twinx()
#     ax2.set_ylabel(r'\textit{d}', labelpad=15, fontsize=36)
    ax2.plot(X_test[image_index], lw=5)

    ax2.grid(False)
    ax2.get_yaxis().set_visible(True)

    ax2.set_yticklabels([])
#     ax.set_xticklabels([])

#     ax.xaxis.set_ticks(np.arange(1, 100, 10))
#     ticks = ['$0$', '$10$', '$20$', '$30$', '$40$', '$50$', '$60$', '$70$', '$80$', '$90$']
# #     lst = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
#     lst = list(np.linspace(0, 100, 10, endpoint=False, dtype=int))
# #     ticks = list(map(str, lst))
#     ax.set_xticklabels(ticks)
#     ax.tick_params(axis='x', direction='out', length=6, width=2, colors='k',
#                grid_color='k', grid_alpha=0.5)

    ax.set_xlabel(r'Sequence Length, $L$', fontsize=28)
    ax2.set_ylabel(r'Attention Weight Percentage', fontsize=28)

    ax.tick_params(which='minor', width=1.25)
    ax.tick_params(which='minor', length=3.5)

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(r'${%d}$'))

    # For the minor ticks, use no labels; default NullFormatter.
#     ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.get_yaxis().set_visible(False)

    plt.axvline(x=99, color="lime", linestyle="dashed", linewidth=2)
  # display the image
    plt.title(rf"Predicted Class: {class_names[desired_class]} with Probability = {results[image_index][prediction]:.3f}", fontsize=36)
    plt.savefig(f"{asnwd}/notebooks/cams/CAM-{class_names[desired_class]}-{counter}")
    plt.show()
    plt.clf()


def show_maps(desired_class, num_maps):
    '''
    goes through the first 10,000 test images and generates CAMs
    for the first `num_maps`(int) of the `desired_class`(int)
    '''

    counter = 0

    if desired_class > (len(class_names) - 1):
        print("please choose a class between 0 and {len(class_names) - 1}")

    # go through the first 10000 images
    for i in range(0,1000):
        # break if we already displayed the specified number of maps
        if counter == num_maps:
            break

        # images that match the class will be shown
        random_sample = np.random.choice(len(results), 1)[0]
        if np.argmax(results[random_sample]) == desired_class:
#         if np.argmax(results[i]) == desired_class:
#             counter += 1
            sc = show_cam(random_sample, desired_class, counter)
            if sc is not None:
                continue
            else:
                counter += 1


# show_maps(desired_class=class_names.index("SNIa"), num_maps=1)
