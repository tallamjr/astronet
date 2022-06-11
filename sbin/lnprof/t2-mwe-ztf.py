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


def extract_fields(alert: Dict):
    # Get flux and error
    magpsf = extract_field(alert, "magpsf")
    sigmapsf = extract_field(alert, "sigmapsf")

    jd = extract_field(alert, "jd")

    # For rescaling dates to start at 0 --> 30
    # dates = np.array([jd[0] - i for i in jd])

    # FINK candidate ID (int64)
    candid = alert["candid"]

    # filter bands
    fid = extract_field(alert, "fid")

    return (magpsf, sigmapsf, jd, candid, fid)


def check_size(filepath):
    du = subprocess.run(
        f"du -sh {filepath} | awk '{{print $1}}'",
        check=True,
        capture_output=True,
        shell=True,
        text=True,
    ).stdout
    return du


def zippify_tflite(tflite_file_path):

    zipped_name = f"{asnwd}/sbin/lnprof/{tflite_file_path}.zip"

    with zipfile.ZipFile(
        zipped_name,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.write(tflite_file_path)

    return zipped_name


@profile
def get_model(model_name: str = "model-23057-1642540624-0.1.dev963+g309c9d8"):
    # Load original keras model
    model_path = f"{asnwd}/astronet/t2/models/plasticc/{model_name}"

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"WeightedLogLoss": WeightedLogLoss()},
        compile=False,
    )

    return model


@profile
def get_compressed_model(model_name: str = "23057-1642540624-0.1.dev963+g309c9d8"):
    # Load compressed clustered keras model i.e. was keras model but saved as .zip file on disk
    model_path = f"{asnwd}/astronet/t2/models/plasticc/{model_name}"

    with zipfile.ZipFile(f"{model_path}.zip", mode="r") as archive:
        for file in archive.namelist():
            archive.extract(file, model_path)

    ccmodel = tf.keras.models.load_model(
        model_path,
        custom_objects={"WeightedLogLoss": WeightedLogLoss()},
        compile=False,
    )
    return ccmodel


@profile
def get_compressed_to_lite_model(
    model_name: str = "23057-1642540624-0.1.dev963+g309c9d8",
):
    # Load compressed clustered model and convert it to a TFLite model, i.e. was keras model but
    # saved as .zip file on disk
    model_path = f"{asnwd}/astronet/t2/models/plasticc/{model_name}"

    with zipfile.ZipFile(f"{model_path}.zip", mode="r") as archive:
        for file in archive.namelist():
            archive.extract(file, model_path)

    cc2lmodel = LiteModel.from_saved_model(model_path)

    return cc2lmodel


@profile
def get_lite_model(
    model_name: str = "23057-1642540624-0.1.dev963+g309c9d8", tflite_file_path=None
):
    # Load clustered model TFLite model, i.e. a .tflife model/file on disk
    model_path = f"{asnwd}/sbin/lnprof/clustered_stripped_fink_model.tflite"
    clmodel = LiteModel.from_file(model_path=model_path)

    return clmodel


@profile
def get_quantized_lite_model(
    model_name: str = "23057-1642540624-0.1.dev963+g309c9d8", tflite_file_path=None
):
    # Load clustered model TFLite model, i.e. a .tflife model/file on disk
    model_path = f"{asnwd}/sbin/lnprof/clustered_stripped_fink_model_quantized.tflite"
    cqlmodel = LiteModel.from_file(model_path=model_path)

    return cqlmodel


@profile
def get_compressed_lite_model(
    model_name: str = "23057-1642540624-0.1.dev963+g309c9d8", tflite_file_path=None
):
    # Load compressed clustered model TFLite model, i.e. was .tflife model/file but saved as .zip
    # file on disk
    model_path = f"{asnwd}/sbin/lnprof/__clustered_stripped_fink_model.tflite"

    with zipfile.ZipFile(f"{model_path}.zip", mode="r") as archive:
        for file in archive.namelist():
            archive.extract(file, model_path)
    # zipfile.ZipFile('hello.zip', mode='w').write("hello.csv")
    cclmodel = LiteModel.from_file(
        model_path=f"{model_path}/clustered_stripped_fink_model.tflite"
    )

    return cclmodel


# @profile
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


if __name__ == "__main__":

    print(astronet.__version__)
    print(astronet.__file__)

    # psdf = ps.read_parquet(f"{asnwd}/data/ztf/alerts.parquet")
    psdf = ps.read_parquet(f"{asnwd}/data/ztf/sample.parquet")

    SEED = 2
    random.seed(SEED)
    r = random.randint(SEED, len(psdf))

    print(r)
    alert = psdf.iloc[r]
    print(alert.head())
    alert = alert.to_dict()

    magpsf, sigmapsf, jd, candid, fid = extract_fields(alert)

    # ORIGINAL MODEL
    model = get_model()

    # ORIGINAL MODEL
    model = get_model()

    # COMPRESSED CLUSTERED-STRIPPED MODEL
    model_name = "tinho/compressed_clustered_stripped_fink_model"
    ccmodel = get_compressed_model(model_name)

    # CLUSTERED-STRIPPED MODEL --> TFLITE
    # model_name = "tinho/clustered_stripped_fink_model"
    # model_path = f"{asnwd}/astronet/t2/models/plasticc/{model_name}"
    # c2lmodel = LiteModel.from_saved_model(model_path)

    # COMPRESSED CLUSTERED-STRIPPED MODEL --> TFLITE
    cc2lmodel = get_compressed_to_lite_model(model_name)

    # CLUSTERED-STRIPPED TFLITE MODEL
    clmodel = get_lite_model()

    # CLUSTERED-STRIPPED QUANTIZED TFLITE MODEL
    clmodel = get_quantized_lite_model()

    # COMPRESSED CLUSTERED-STRIPPED TFLITE  MODEL
    model_name = "tinho/compressed_clcmodel"
    model_path = f"{asnwd}/astronet/t2/models/plasticc/{model_name}"
    tflite_file_path = "clustered_stripped_fink_model.tflite"

    # ccmodel = LiteModel.from_saved_model(model_path, tflite_file_path=tflite_file_path)
    # cclmodel_zipped = zippify_tflite(tflite_file_path)
    # print(
    #     f"COMPRESSED TFLITE CLUSTERED-STRIPPED MODEL SIZE: {check_size(cclmodel_zipped)}"
    # )
    cclmodel = get_compressed_lite_model()

    # t2_probs(candid, jd, fid, magpsf, sigmapsf, model=model, prettyprint=True)    # t2-mwe-ztf-original-model.lnprofile
    # t2_probs(candid, jd, fid, magpsf, sigmapsf, model=cmodel, prettyprint=True)   # t2-mwe-ztf-compressed-model.lnprofile
    # t2_probs(candid, jd, fid, magpsf, sigmapsf, model=lmodel, prettyprint=True)   # t2-mwe-ztf-clustered-tflite-model.lnprofile
    # t2_probs(candid, jd, fid, magpsf, sigmapsf, model=clmodel, prettyprint=True)  # t2-mwe-ztf-compressed-clustered-tflite-model.lnprofile

    # t2_probs(candid, jd, fid, magpsf, sigmapsf, model=cclmodel, prettyprint=True)   # t2-mwe-ztf-compressed-tflite-model.lnprofile
    # t2_probs(
    #     candid, jd, fid, magpsf, sigmapsf, model=clmodel, prettyprint=True
    # )  # t2-mwe-ztf-compressed-tflite-model.lnprofile
