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
from astronet.tests.reg.get_models import (
    get_clustered_model,
    get_compressed_clustered_model,
    get_compressed_clustered_pruned_model,
    get_compressed_model,
    get_model,
    get_pruned_model,
    get_quantized_tflite_from_file,
    get_tflite_from_file,
)
from astronet.tinho.compress import (
    inspect_model,
    print_clusters,
    print_sparsity,
)
from astronet.utils import astronet_logger

log = astronet_logger(__file__)

warnings.filterwarnings("ignore")

np_config.enable_numpy_behavior()
# flake8: noqa: C901


@profile
def predict_original_model(X_test, wloss):
    # ORIGINAL T2 MODEL ON GR-noZ
    # BASELINE
    model = get_model()
    y_preds = model.predict(X_test)
    log.info(
        f"BASELINE :ORIGINAL T2 MODEL ON GR-noZ LL-Test: {wloss(y_test, y_preds).numpy():.3f}"
    )
    log.info(
        f"\n{print_sparsity(model)}\n{print_clusters(model)}\n{inspect_model(model)}"
    )


@profile
def predict_compressed_model(X_test, wloss):
    # COMPRESSED MODEL, aka COMPRESSED T2
    # BASELINE + HUFFMAN
    model = get_compressed_model()
    y_preds = model.predict(X_test)
    log.info(
        f"BASELINE + HUFFMAN, aka COMPRESSED T2 LL-Test: {wloss(y_test, y_preds).numpy():.3f}"
    )
    log.info(
        f"\n{print_sparsity(model)}\n{print_clusters(model)}\n{inspect_model(model)}"
    )


@profile
def predict_clustered_model(X_test, wloss):
    # CLUSTERED-STRIPPED MODEL, aka TINHO
    # CLUSTERING
    model = get_clustered_model()
    y_preds = model.predict(X_test)
    log.info(f"CLUSTERING, aka TINHO LL-Test: {wloss(y_test, y_preds).numpy():.3f}")
    log.info(
        f"\n{print_sparsity(model)}\n{print_clusters(model)}\n{inspect_model(model)}"
    )


@profile
def predict_pruned_model(X_test, wloss):
    # PRUNED-STRIPPED MODEL
    # CLUSTERING + PRUNING
    model = get_pruned_model()
    y_preds = model.predict(X_test)
    log.info(f"PRUNING LL-Test: {wloss(y_test, y_preds).numpy():.3f}")
    log.info(
        f"\n{print_sparsity(model)}\n{print_clusters(model)}\n{inspect_model(model)}"
    )


@profile
def predict_compressed_clustered_model(X_test, wloss):
    # COMPRESSED CLUSTERED-STRIPPED MODEL, aka COMPRESSED TINHO
    # CLUSTERING + HUFFMAN
    model = get_compressed_clustered_model()
    y_preds = model.predict(X_test)
    log.info(
        f"CLUSTERING + HUFFMAN, aka COMPRESSED TINHO LL-Test: {wloss(y_test, y_preds).numpy():.3f}"
    )
    log.info(
        f"\n{print_sparsity(model)}\n{print_clusters(model)}\n{inspect_model(model)}"
    )


@profile
def predict_compressed_clustered_pruned_model(X_test, wloss):
    # COMPRESSED CLUSTERED-PRUNED-STRIPPED MODEL, aka COMPRESSED TINHO
    # CLUSTERING + PRUNING + HUFFMAN
    model = get_compressed_clustered_pruned_model()
    y_preds = model.predict(X_test)
    log.info(
        f"CLUSTERING + PRUNING + HUFFMAN LL-Test: {wloss(y_test, y_preds).numpy():.3f}"
    )
    log.info(
        f"\n{print_sparsity(model)}\n{print_clusters(model)}\n{inspect_model(model)}"
    )


@profile
def predict_saved_clustered_tflite_model(X_test, wloss):
    # SAVED TFLITE CLUSTERED-STRIPPED MODEL, .tflife FILE
    # CLUSTERING-FLATBUFFER
    # Load clustered model TFLite model, i.e. a .tflife model/file on disk
    model_path = f"{asnwd}/astronet/tinho/models/plasticc/model-GR-noZ-28341-1654269564-0.5.1.dev73+g70f85f8-LL0.836.tflite"
    model = get_tflite_from_file(model_path)
    y_preds = model.predict(X_test)
    log.info(
        f"CLUSTERING-FLATBUFFER MODEL LL-Test: {wloss(y_test, y_preds).numpy():.3f}"
    )


@profile
def predict_saved_clustered_quantized_tflite_model(X_test, wloss):
    # SAVED QUANTIZED TFLITE CLUSTERED-STRIPPED MODEL, .tflife FILE
    # CLUSTERING-FLATBUFFER + QUANTIZATION
    # Load clustered model TFLite model, i.e. a .tflife model/file on disk
    # model_path = f"{asnwd}/sbin/lnprof/clustered_stripped_fink_model_quantized.tflite"
    model_path = f"{asnwd}/astronet/tinho/models/plasticc/quantized-model-GR-noZ-28341-1654269564-0.5.1.dev73+g70f85f8-LL0.836.tflite"
    model = get_quantized_tflite_from_file(model_path)
    y_preds = model.predict(X_test)
    log.info(
        f"CLUSTERING-FLATBUFFER + QUANTIZATION LL-Test: {wloss(y_test, y_preds).numpy():.3f}"
    )


# ------------------------------------------------------------------- #


@profile
def predict_clustered_tflite_model(X_test, wloss):
    # CLUSTERED-STRIPPED MODEL (TINHO), TFLITE INTERPRETER
    # CLUSTERING-FLATBUFFER CONVERSION
    model_path = f"{asnwd}/astronet/t2/models/plasticc/{model_name}"
    lmodel = get_tflite_from_saved_model(model_path)
    y_preds = model.predict(X_test)
    log.info(
        f"CLUSTERING-FLATBUFFER CONVERSION LL-Test: {wloss(y_test, y_preds).numpy():.3f}"
    )


@profile
def predict_compressed_clustered_tflite_model(X_test, wloss):
    # COMPRESSED CLUSTERED-STRIPPED MODEL (TINHO), TFLITE INTERPRETER
    # CLUSTERING-FLATBUFFER + HUFFMAN
    # TODO: Update model_name
    model_name = "tinho/compressed_clustered_stripped_fink_model"
    clmodel = get_compressed_lite_model(model_name)
    y_preds = model.predict(X_test)
    log.info(
        f"CLUSTERING-FLATBUFFER + HUFFMAN MODEL LL-Test: {wloss(y_test, y_preds).numpy():.3f}"
    )


if __name__ == "__main__":
    """
    Test the load and inference times of models saved in different formats, specifcally to compare
    the best version with the best overall latency, yet with the best score.
    """

    log.info(astronet.__version__)
    log.info(astronet.__file__)

    LOAD_ONLY = True

    if LOAD_ONLY:

        _ = get_tflite_from_file(
            f"{asnwd}/astronet/tinho/models/plasticc/model-GR-noZ-28341-1654269564-0.5.1.dev73+g70f85f8-LL0.836.tflite"
        )

        _ = get_quantized_tflite_from_file(
            f"{asnwd}/astronet/tinho/models/plasticc/quantized-model-GR-noZ-28341-1654269564-0.5.1.dev73+g70f85f8-LL0.836.tflite"
        )

    else:
        X_test = np.load(
            f"{asnwd}/data/plasticc/test_set/infer/X_test.npy",
        )
        y_test = np.load(
            f"{asnwd}/data/plasticc/test_set/infer/y_test.npy",
        )

        log.info(f"X_TEST: {X_test.shape}, Y_TEST: {y_test.shape}")

        # Only trained on red, green filters {r, g}
        X_test = X_test[:, :, 0:3:2]

        log.info("Running predictions")
        wloss = WeightedLogLoss()

        # BASELINE
        predict_original_model(X_test, wloss)
        # BASELINE + HUFFMAN
        predict_compressed_model(X_test, wloss)

        # CLUSTERING
        predict_clustered_model(X_test, wloss)
        # CLUSTERING + HUFFMAN
        predict_compressed_clustered_model(X_test, wloss)

        # CLUSTERING + PRUNING
        predict_pruned_model(X_test, wloss)
        # CLUSTERING + PRUNING + HUFFMAN
        predict_compressed_clustered_pruned_model(X_test, wloss)

        # CLUSTERING-FLATBUFFER
        predict_saved_clustered_tflite_model(X_test, wloss)
        # CLUSTERING-FLATBUFFER + QUANTIZATION
        predict_saved_clustered_quantized_tflite_model(X_test, wloss)