import random
import warnings
import zipfile

import pyspark.pandas as ps
from fink_client.visualisation import extract_field

import astronet
from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd

warnings.filterwarnings("ignore")

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from fink_utils.photometry.conversion import mag2fluxcal_snana
from tensorflow.python.ops.numpy_ops import np_config

from astronet.metrics import WeightedLogLoss

np_config.enable_numpy_behavior()


class LiteModel:
    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_saved_model(cls, model_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        ]
        converter.experimental_enable_resource_variables = True
        converter.experimental_new_converter = True
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i : i + 1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out


def get_model(model_name: str = "model-23057-1642540624-0.1.dev963+g309c9d8"):
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

    # predict_original_model(X_test, wloss)
    # predict_clustered_tflite_model(X_test, wloss)

    # predict_saved_clustered_tflite_model(X_test, wloss)
    predict_saved_clustered_quantized_tflite_model(X_test, wloss)

    # predict_compressed_clustered_tflite_model(X_test, wloss)
    # predict_compressed_clustered_model(X_test, wloss)
