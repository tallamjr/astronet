import random
import subprocess
import warnings
import zipfile

import pyspark.pandas as ps
from fink_client.visualisation import extract_field
from sklearn.preprocessing import robust_scale as rs

import astronet
from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd

# %load -r 37-255 processor.py
from astronet.preprocess import generate_gp_all_objects, robust_scale

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
    def from_saved_model(cls, model_path, tflite_file_path=None):
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
        ]
        converter.experimental_enable_resource_variables = True
        converter.experimental_new_converter = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        if tflite_file_path is not None:
            with open(tflite_file_path, "wb") as f:
                f.write(tflite_model)

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


def get_lite_model(
    model_name: str = "23057-1642540624-0.1.dev963+g309c9d8", tflite_file_path=None
):
    # Load clustered model TFLite model, i.e. a .tflife model/file on disk
    model_path = f"{asnwd}/sbin/lnprof/clustered_stripped_fink_model.tflite"
    clmodel = LiteModel.from_file(model_path=model_path)

    return clmodel


def get_compressed_lite_model(
    model_name: str = "23057-1642540624-0.1.dev963+g309c9d8", tflite_file_path=None
):
    # Load compressed clustered model TFLite model, i.e. was .tflife model/file but saved as .zip
    # file on disk
    model_path = f"{asnwd}/sbin/lnprof/__clustered_stripped_fink_model.tflite"

    with zipfile.ZipFile(f"{model_path}.zip", mode="r") as archive:
        for file in archive.namelist():
            archive.extract(file, model_path)

    cclmodel = LiteModel.from_file(
        model_path=f"{model_path}/clustered_stripped_fink_model.tflite"
    )

    return cclmodel


def get_clustered_model_to_lite(
    tflite_file_path="clustered_stripped_fink_model_quantized.tflite",
):

    model_path = (
        f"{asnwd}/astronet/t2/models/plasticc/tinho/clustered_stripped_fink_model"
    )
    c2lmodel = LiteModel.from_saved_model(model_path, tflite_file_path=tflite_file_path)

    return c2lmodel


if __name__ == "__main__":

    model = get_lite_model()
    model = get_compressed_lite_model()
    model = get_clustered_model_to_lite()
