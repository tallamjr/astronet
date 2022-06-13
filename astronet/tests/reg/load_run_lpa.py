import random
import warnings
import zipfile

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd

import astronet

warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from astronet.metrics import WeightedLogLoss
from astronet.tinho.lite import LiteModel
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def get_model(
    model_name: str = "model-GR-noZ-23057-1642540624-0.1.dev963+g309c9d8-LL0.968",
):
    # Load pre-trained model
    model_path = f"{asnwd}/astronet/t2/models/plasticc/{model_name}"

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"WeightedLogLoss": WeightedLogLoss()},
        compile=False,
    )

    return model


def get_compressed_lite_model(model_name: str = "23057-1642540624-0.1.dev963+g309c9d8"):
    # Load pre-trained model
    model_path = f"{asnwd}/astronet/t2/models/plasticc/{model_name}"

    with zipfile.ZipFile(f"{model_path}.zip", mode="r") as archive:
        for file in archive.namelist():
            archive.extract(file, model_path)

    lmodel = LiteModel.from_saved_model(model_path)

    return lmodel


def get_compressed_model(model_name: str = "23057-1642540624-0.1.dev963+g309c9d8"):
    # Load pre-trained model
    model_path = f"{asnwd}/astronet/t2/models/plasticc/{model_name}"

    with zipfile.ZipFile(f"{model_path}.zip", mode="r") as archive:
        for file in archive.namelist():
            archive.extract(file, model_path)

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"WeightedLogLoss": WeightedLogLoss()},
        compile=False,
    )
    return model


@profile
def predict_original_model(X_test, wloss):
    # ORIGINAL MODEL
    model = get_model()
    y_preds = model.predict(X_test)
    print(f"ORIGINAL MODEL LL-Test: {wloss(y_test, y_preds).numpy():.2f}")


@profile
def predict_compressed_clustered_model(X_test, wloss):
    # COMPRESSED CLUSTERED-STRIPPED MODEL
    model_name = "tinho/compressed_clustered_stripped_fink_model"
    cmodel = get_compressed_model(model_name)
    y_preds = cmodel.predict(X_test)
    print(
        f"COMPRESSED CLUSTERED-STRIPPED MODEL LL-Test: {wloss(y_test, y_preds).numpy():.2f}"
    )


@profile
def predict_clustered_tflite_model(X_test, wloss):
    # TFLITE CLUSTERED-STRIPPED MODEL
    model_name = "tinho/clustered_stripped_fink_model"
    model_path = f"{asnwd}/astronet/t2/models/plasticc/{model_name}"
    lmodel = LiteModel.from_saved_model(model_path)
    y_preds = lmodel.predict(X_test)
    print(
        f"TFLITE CLUSTERED-STRIPPED MODEL LL-Test: {wloss(y_test, y_preds).numpy():.2f}"
    )


@profile
def predict_compressed_clustered_tflite_model(X_test, wloss):
    # TFLITE COMPRESSED CLUSTERED-STRIPPED MODEL
    model_name = "tinho/compressed_clustered_stripped_fink_model"
    clmodel = get_compressed_lite_model(model_name)
    y_preds = clmodel.predict(X_test)
    print(
        f"TFLITE COMPRESSED CLUSTERED-STRIPPED MODEL LL-Test: {wloss(y_test, y_preds).numpy():.2f}"
    )


@profile
def predict_saved_clustered_tflite_model(X_test, wloss):
    # SAVED TFLITE COMPRESSED CLUSTERED-STRIPPED MODEL
    # Load clustered model TFLite model, i.e. a .tflife model/file on disk
    model_path = f"{asnwd}/sbin/lnprof/clustered_stripped_fink_model.tflite"
    clmodel = LiteModel.from_file(model_path=model_path)
    y_preds = clmodel.predict(X_test)
    print(
        f"SAVED TFLITE COMPRESSED CLUSTERED-STRIPPED MODEL LL-Test: {wloss(y_test, y_preds).numpy():.2f}"
    )


@profile
def predict_saved_clustered_quantized_tflite_model(X_test, wloss):
    # SAVED TFLITE COMPRESSED CLUSTERED-STRIPPED MODEL
    # Load clustered model TFLite model, i.e. a .tflife model/file on disk
    # model_path = f"{asnwd}/sbin/lnprof/clustered_stripped_fink_model_quantized.tflite"
    model_path = f"{asnwd}/sbin/lnprof/quantized.tflite"
    clmodel = LiteModel.from_file(model_path=model_path)
    y_preds = clmodel.predict(X_test)
    print(
        f"SAVED QUANTIZED TFLITE COMPRESSED CLUSTERED-STRIPPED MODEL LL-Test: {wloss(y_test, y_preds).numpy():.2f}"
    )


if __name__ == "__main__":
    """
    Test the load and inference times of models saved in different formats, specifcally to compare
    the best version with the best overall latency, yet with the best score.
    """

    print(astronet.__version__)
    print(astronet.__file__)

    X_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/X_test.npy",
    )
    y_test = np.load(
        f"{asnwd}/data/plasticc/test_set/infer/y_test.npy",
    )

    print(f"X_TEST: {X_test.shape}, Y_TEST: {y_test.shape}")

    # Only trained on red, green filters {r, g}
    X_test = X_test[:, :, 0:3:2]

    print("Running predictions")
    wloss = WeightedLogLoss()

    predict_original_model(X_test, wloss)
    predict_clustered_tflite_model(X_test, wloss)

    predict_saved_clustered_tflite_model(X_test, wloss)
    predict_saved_clustered_quantized_tflite_model(X_test, wloss)

    predict_compressed_clustered_tflite_model(X_test, wloss)
    predict_compressed_clustered_model(X_test, wloss)
