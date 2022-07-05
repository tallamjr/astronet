# Copyright 2020 - 2022
# Author: Tarek Allam Jr.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import tempfile
import warnings
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

import astronet
from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.metrics import WeightedLogLoss
from astronet.tinho.compress import (
    inspect_model,
    print_clusters,
    print_sparsity,
)
from astronet.tinho.lite import LiteModel
from astronet.utils import astronet_logger

log = astronet_logger(__file__)

warnings.filterwarnings("ignore")

np_config.enable_numpy_behavior()
# flake8: noqa: C901


def get_model(
    model_name: str = "model-GR-noZ-23057-1642540624-0.1.dev963+g309c9d8-LL0.968",
):
    # Load pre-trained original t2 model
    model_path = f"{asnwd}/astronet/t2/models/plasticc/{model_name}"

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"WeightedLogLoss": WeightedLogLoss()},
        compile=False,
    )

    return model


def get_compressed_model(
    model_name: str = "model-GR-noZ-23057-1642540624-0.1.dev963+g309c9d8-LL0.968",
):
    # Load pre-trained zipped original t2 model
    model_path = f"{asnwd}/astronet/t2/models/plasticc/{model_name}"

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(f"{model_path}.zip", mode="r") as archive:
            for file in archive.namelist():
                archive.extract(file, tmpdir)

        model = tf.keras.models.load_model(
            f"{tmpdir}/{model_name}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

    return model


def get_compressed_convert_to_lite(
    model_name: str = "model-GR-noZ-23057-1642540624-0.1.dev963+g309c9d8-LL0.968",
):
    # Load pre-trained model
    model_path = f"{asnwd}/astronet/t2/models/plasticc/{model_name}"

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(f"{model_path}.zip", mode="r") as archive:
            for file in archive.namelist():
                archive.extract(file, tmpdir)

        lmodel = get_tflite_from_saved_model(f"{tmpdir}/{model_name}")

    return lmodel


def get_clustered_model(
    model_name: str = "model-GR-noZ-28341-1654269564-0.5.1.dev73+g70f85f8-LL0.836",
):
    # Load pre-trained original t2 model
    model_path = f"{asnwd}/astronet/tinho/models/plasticc/{model_name}"

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"WeightedLogLoss": WeightedLogLoss()},
        compile=False,
    )

    return model


def get_compressed_clustered_model(
    model_name: str = "model-GR-noZ-28341-1654269564-0.5.1.dev73+g70f85f8-LL0.836",
):
    # Load pre-trained model
    model_path = f"{asnwd}/astronet/tinho/models/plasticc/{model_name}"

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(f"{model_path}.zip", mode="r") as archive:
            for file in archive.namelist():
                archive.extract(file, tmpdir)

        model = tf.keras.models.load_model(
            f"{tmpdir}/{model_name}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

    return model


def get_tflite_from_file(
    model_path: str = f"{asnwd}/astronet/tinho/models/plasticc/model-GR-noZ-28341-1654269564-0.5.1.dev73+g70f85f8-LL0.836.tflite",
):
    return LiteModel.from_file(model_path=model_path)


def get_quantized_tflite_from_file(
    model_path: str = f"{asnwd}/astronet/tinho/models/plasticc/quantized-model-GR-noZ-28341-1654269564-0.5.1.dev73+g70f85f8-LL0.836.tflite",
):
    return LiteModel.from_file(model_path=model_path)


def get_tflite_from_saved_model(model_path: str):
    return LiteModel.from_saved_model(model_path=model_path)


def get_pruned_model(
    model_name: str = "model-GR-noZ-9903651-1652692724-0.5.1.dev24+gb7cd783.d20220516-STRIPPED-PRUNED",
):
    # Load pre-trained original t2 model
    model_path = f"{asnwd}/astronet/tinho/models/plasticc/{model_name}"

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"WeightedLogLoss": WeightedLogLoss()},
        compile=False,
    )

    return model


def get_compressed_clustered_pruned_model(
    model_name: str = "model-GR-noZ-9903651-1652692724-0.5.1.dev24+gb7cd783.d20220516-STRIPPED-PRUNED",
):
    # Load pre-trained model
    model_path = f"{asnwd}/astronet/tinho/models/plasticc/{model_name}"

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(f"{model_path}.zip", mode="r") as archive:
            for file in archive.namelist():
                archive.extract(file, tmpdir)

        model = tf.keras.models.load_model(
            f"{tmpdir}/{model_name}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

    return model
