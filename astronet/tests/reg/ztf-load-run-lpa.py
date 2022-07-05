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

import random
import subprocess
import warnings
import zipfile

import pyspark.pandas as ps
from fink_utils.data.utils import format_data_as_snana
from sklearn.preprocessing import robust_scale as rs

import astronet
from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.dutils import extract_fields
from astronet.preprocess import generate_gp_all_objects

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

import tempfile

from astronet.metrics import WeightedLogLoss
from astronet.tests.reg.get_models import (
    __get_clustered_model,
    __get_compressed_clustered_model,
    __get_compressed_clustered_pruned_model,
    __get_compressed_model,
    __get_model,
    __get_pruned_model,
    __get_quantized_tflite_from_file,
    __get_tflite_from_file,
)
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


@profile
def t2_probs(
    candid: np.int64,
    jd: np.ndarray,
    fid: np.ndarray,
    magpsf: np.int64,
    sigmapsf: np.int64,
    model: tf.keras.Model,
    prettyprint=None,
) -> Dict:
    """Compute probabilities of alerts in relation to PLAsTiCC classes using the Time-Series
    Transformer model

    Parameters
    ----------
    candid: np.int64,
        Candidate IDs
    jd: np.ndarray,
        JD times (float)
    fid: np.ndarray,
        Filter IDs (int)
    magpsf: np.ndarray,
        Magnitude from PSF-fit photometry
    sigmapsf: np.ndarray,
        1-sigma error of PSF-fit
    model: tensorflow.python.keras.saving.saved_model.load.T2Model
        Pre-compiled T2 model

    Returns
    ----------
    probabilities: dict
        Dict containing np.array of float probabilities

    Examples
    ----------
    >>> import pyspark.pandas as ps
    >>> psdf = ps.read_parquet('sample.parquet')
    >>> import random
    >>> r = random.randint(0,len(psdf))
    >>> alert = psdf.iloc[r]
    >>> print(alert.head())
    candid                                     1786552611115010001
    schemavsn                                                  3.3
    publisher                                                 Fink
    objectId                                          ZTF18aaqfhlj
    candidate    (2459541.0526157, 2, 1786552611115, 19.1966800...
    Name: 221, dtype: object
    >>> alert = alert.to_dict()

    >>> from fink_client.visualisation import extract_field
    # Get flux and error
    >>> magpsf = extract_field(alert, 'magpsf')
    >>> sigmapsf = extract_field(alert, 'sigmapsf')

    >>> jd = extract_field(alert, "jd")

    # For rescaling dates to start at 0 --> 30
    # dates = np.array([jd[0] - i for i in jd])

    # FINK candidate ID (int64)
    >>> candid = alert["candid"]

    # filter bands
    >>> fid = extract_field(alert, "fid")

    >>> model_name = "23057-1642540624-0.1.dev963+g309c9d8"
    >>> model = get_model(model_name=model_name)

    >>> t2_probs(candid, jd, fid, magpsf, sigmapsf, model)
    {
        "AGN": 0.0,
        "EB": 0.017,
        "KN": 0.0,
        "M-dwarf": 0.891,
        "Mira": 0.002,
        "RRL": 0.004,
        "SLSN-I": 0.0,
        "SNII": 0.078,
        "SNIa": 0.001,
        "SNIa-91bg": 0.006,
        "SNIax": 0.001,
        "SNIbc": 0.001,
        "TDE": 0.0,
        "mu-Lens-Single": 0.0
    }
    """

    ZTF_FILTER_MAP = {1: "ztfg", 2: "ztfr", 3: "ztfi"}

    ZTF_PB_WAVELENGTHS = {
        "ztfg": 4804.79,
        "ztfr": 6436.92,
        "ztfi": 7968.22,
    }

    # Rescale dates to _start_ at 0
    dates = np.array([jd[0] - i for i in jd])

    mjd, flux, flux_error, filters = ([] for i in range(4))

    # Loop over each filter
    filter_color = ZTF_FILTER_MAP
    for filt in filter_color.keys():
        mask = np.where(fid == filt)[0]

        # Skip if no data
        if len(mask) == 0:
            continue

        maskNotNone = magpsf[mask] != None
        mjd.append(dates[mask][maskNotNone])
        flux.append(magpsf[mask][maskNotNone])
        flux_error.append(sigmapsf[mask][maskNotNone])
        filters.append(filt)

    df_tmp = pd.DataFrame.from_dict(
        {
            "mjd": mjd,
            "object_id": candid,
            "flux": flux,
            "flux_error": flux_error,
            "filters": filters,
        }
    )

    df_tmp = df_tmp.apply(pd.Series.explode).reset_index()

    # Re-compute flux and flux error
    data = [
        mag2fluxcal_snana(*args)
        for args in zip(df_tmp["flux"].explode(), df_tmp["flux_error"].explode())
    ]
    flux, error = np.transpose(data)

    # make a Pandas DataFrame with exploded series
    pdf = pd.DataFrame.from_dict(
        {
            "filter": df_tmp["filters"].replace({1: "ztfg", 2: "ztfr"}),
            "flux": flux,
            "flux_error": error,
            "mjd": df_tmp["mjd"],
            "object_id": df_tmp["object_id"],
        }
    )

    pdf = pdf.filter(["object_id", "mjd", "flux", "flux_error", "filter"])
    # pdf = pdf.dropna()
    # pdf = pdf.reset_index()

    if not isinstance(candid, list):
        object_list = [candid]
    df_gp_mean = generate_gp_all_objects(
        object_list, pdf, pb_wavelengths=ZTF_PB_WAVELENGTHS
    )

    cols = set(list(ZTF_PB_WAVELENGTHS.keys())) & set(df_gp_mean.columns)
    # robust_scale(df_gp_mean, cols)
    X = df_gp_mean[cols]
    X = rs(X)
    X = np.asarray(X).astype("float32")
    X = np.expand_dims(X, axis=0)

    y_preds = model.predict(X)
    # y_preds = model(X)  # t2_probs(candid, jd, fid, magpsf, sigmapsf, model=cmodel, prettyprint=True) --> t2-mwe-ztf-compressed-model-nopredict.lnprofile

    class_names = [
        "mu-Lens-Single",
        "TDE",
        "EB",
        "SNII",
        "SNIax",
        "Mira",
        "SNIbc",
        "KN",
        "M-dwarf",
        "SNIa-91bg",
        "AGN",
        "SNIa",
        "RRL",
        "SLSN-I",
    ]

    keys = class_names
    values = y_preds.tolist()
    predictions = dict(zip(keys, values[0]))

    if prettyprint is not None:
        import json

        print(
            json.dumps(
                json.loads(
                    json.dumps(predictions), parse_float=lambda x: round(float(x), 3)
                ),
                indent=4,
                sort_keys=True,
            )
        )

    return predictions


@profile
def predict_t2_baseline():
    # BASELINE
    model = __get_model()
    t2_probs(candid, jd, fid, magpsf, sigmapsf, model=model, prettyprint=True)


@profile
def predict_t2_baseline_huffman():
    # BASELINE + HUFFMAN
    model = __get_compressed_model()
    t2_probs(candid, jd, fid, magpsf, sigmapsf, model=model, prettyprint=True)


@profile
def predict_t2_clustering():
    # CLUSTERING
    model = __get_clustered_model()
    t2_probs(candid, jd, fid, magpsf, sigmapsf, model=model, prettyprint=True)


@profile
def predict_t2_clustering_huffman():
    # CLUSTERING + HUFFMAN
    model = __get_compressed_clustered_model()
    t2_probs(candid, jd, fid, magpsf, sigmapsf, model=model, prettyprint=True)


@profile
def predict_t2_clustering_pruning():
    # CLUSTERING + PRUNING
    model = __get_pruned_model()
    t2_probs(candid, jd, fid, magpsf, sigmapsf, model=model, prettyprint=True)


@profile
def predict_t2_clustering_pruning_huffman():
    # CLUSTERING + PRUNING + HUFFMAN
    model = __get_compressed_clustered_pruned_model()
    t2_probs(candid, jd, fid, magpsf, sigmapsf, model=model, prettyprint=True)


@profile
def predict_t2_clustering_flatbuffer():
    # CLUSTERING-FLATBUFFER
    model = __get_tflite_from_file()
    t2_probs(candid, jd, fid, magpsf, sigmapsf, model=model, prettyprint=True)


@profile
def predict_t2_clustering_flatbuffer_quantization():
    # CLUSTERING-FLATBUFFER + QUANTIZATION
    model = __get_quantized_tflite_from_file()
    t2_probs(candid, jd, fid, magpsf, sigmapsf, model=model, prettyprint=True)


if __name__ == "__main__":

    print(astronet.__version__)
    print(astronet.__file__)

    psdf = ps.read_parquet(f"{asnwd}/data/ztf/sample.parquet")

    SEED = 2
    random.seed(SEED)
    r = random.randint(SEED, len(psdf))

    print(r)
    alert = psdf.iloc[r]
    print(alert.head())
    alert = alert.to_dict()

    magpsf, sigmapsf, jd, candid, fid = extract_fields(alert)

    # BASELINE
    predict_t2_baseline()
    # BASELINE + HUFFMAN
    predict_t2_baseline_huffman()

    # CLUSTERING
    predict_t2_clustering()
    # CLUSTERING + HUFFMAN
    predict_t2_clustering_huffman()

    # CLUSTERING + PRUNING
    predict_t2_clustering_pruning()
    # CLUSTERING + PRUNING + HUFFMAN
    predict_t2_clustering_pruning_huffman()

    # CLUSTERING-FLATBUFFER
    predict_t2_clustering_flatbuffer()
    # CLUSTERING - FLATBUFFER + QUANTIZATION
    predict_t2_clustering_flatbuffer_quantization()
