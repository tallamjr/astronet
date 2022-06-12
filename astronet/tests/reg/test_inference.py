import os
import random as python_random

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.constants import LOCAL_DEBUG
from astronet.metrics import WeightedLogLoss
from astronet.tests.conftest import BATCH_SIZE
from astronet.tinho.lite import LiteModel
from astronet.utils import astronet_logger

log = astronet_logger(__file__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)


@pytest.mark.skipif(
    os.getenv("CI") is not None, reason="Requires large data files. Run locally."
)
class TestInference:
    """Test inference scores of the best performing models to date. This class is also a proxy for
    inference timing of each model"""

    @pytest.mark.parametrize(
        ("architecture", "dataset", "model_name"),
        (
            (
                "atx",
                "plasticc",
                "UGRIZY-wZ-9887359-1641295475-0.1.dev943+gc9bafac.d20220104-LL0.739",
            ),
            (
                "t2",
                "plasticc",
                "UGRIZY-wZ-1619624444-0.1.dev765+g7c90cbb.d20210428-LL0.507",
            ),
            (
                "tinho",
                "plasticc",
                "UGRIZY-wZ-31367-1654360237-0.5.1.dev78+g702e399.d20220604-LL0.450",
            ),
        ),
    )
    def test_inference_UGRIZY_wZ(
        self, architecture, dataset, model_name, get_fixt_UGRIZY_wZ
    ):

        # Previous models were trained using numpy data as the inputs, newer models leverage
        # tf.data.Dataset instead for faster inference. This is a legacy requirment.
        # Fix ValueError of shape mismatch.
        X_test, y_test, Z_test = get_fixt_UGRIZY_wZ

        test_input = [X_test, Z_test]

        test_ds = (
            tf.data.Dataset.from_tensor_slices(
                ({"input_1": test_input[0], "input_2": test_input[1]}, y_test)
            )
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        y_test_ds = (
            tf.data.Dataset.from_tensor_slices(y_test)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        if LOCAL_DEBUG is not None:
            log.info("LOCAL_DEBUG set, reducing dataset size...")
            test_ds = test_ds.take(300)
            y_test_ds = y_test_ds.take(300)

        worker_id = (
            os.environ.get("PYTEST_XDIST_WORKER")
            if "PYTEST_CURRENT_TEST" in os.environ
            else 0
        )
        log.info(f"Data loaded successfully on worker: {worker_id}")

        model = keras.models.load_model(
            f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

        wloss = WeightedLogLoss()
        y_preds = model.predict(test_input)

        y_test = np.concatenate([y for y in y_test_ds], axis=0)

        if architecture == "atx":
            assert wloss(y_test, y_preds).numpy() == pytest.approx(0.739, 0.01)
        if architecture == "t2":
            assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)
        if architecture == "tinho":
            assert wloss(y_test, y_preds).numpy() == pytest.approx(0.450, 0.01)

    @pytest.mark.parametrize(
        ("architecture", "dataset", "model_name"),
        (
            # TODO: ("atx", "plasticc", "XXX"), --> Missing model ID. Need to reproduce 0.929 result
            ("t2", "plasticc", "UGRIZY-noZ-1619802068-0.1.dev779+ga930d1d-LL0.873"),
        ),
    )
    def test_inference_UGRIZY_noZ(
        self, architecture, dataset, model_name, get_fixt_UGRIZY_wZ
    ):

        X_test, y_test, _ = get_fixt_UGRIZY_wZ

        test_input = X_test

        test_ds = (
            tf.data.Dataset.from_tensor_slices((test_input, y_test))
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        y_test_ds = (
            tf.data.Dataset.from_tensor_slices(y_test)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        if LOCAL_DEBUG is not None:
            print("LOCAL_DEBUG set, reducing dataset size...")
            test_ds = test_ds.take(300)
            y_test_ds = y_test_ds.take(300)

        y_test = np.concatenate([y for y in y_test_ds], axis=0)
        x_test = np.concatenate([x for x, y in test_ds], axis=0)

        model = keras.models.load_model(
            f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

        wloss = WeightedLogLoss()
        y_preds = model.predict(x_test)

        if architecture == "atx":
            assert wloss(y_test, y_preds).numpy() == pytest.approx(0.969, 0.01)
        if architecture == "t2":
            assert wloss(y_test, y_preds).numpy() == pytest.approx(0.873, 0.01)

    @pytest.mark.parametrize(
        ("architecture", "dataset", "model_name"),
        (
            (
                "atx",
                "plasticc",
                "GR-noZ-206145-1644662345-0.3.1.dev36+gfd02ace-LL0.969",
            ),
            (
                "t2",
                "plasticc",
                "GR-noZ-23057-1642540624-0.1.dev963+g309c9d8-LL0.968",
            ),
            (
                "tinho",
                "plasticc",
                "GR-noZ-28341-1654269564-0.5.1.dev73+g70f85f8-LL0.836",
            ),
        ),
    )
    def test_inference_GR_noZ(
        self, architecture, dataset, model_name, get_fixt_UGRIZY_wZ
    ):

        X_test, y_test, _ = get_fixt_UGRIZY_wZ
        X_test = X_test[:, :, 0:3:2]

        test_input = X_test

        test_ds = (
            tf.data.Dataset.from_tensor_slices((test_input, y_test))
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        y_test_ds = (
            tf.data.Dataset.from_tensor_slices(y_test)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        if LOCAL_DEBUG is not None:
            print("LOCAL_DEBUG set, reducing dataset size...")
            test_ds = test_ds.take(300)
            y_test_ds = y_test_ds.take(300)

        y_test = np.concatenate([y for y in y_test_ds], axis=0)

        model = keras.models.load_model(
            f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

        wloss = WeightedLogLoss()
        y_preds = model.predict(test_ds)

        if architecture == "atx":
            assert wloss(y_test, y_preds).numpy() == pytest.approx(0.969, 0.01)
        if architecture == "t2":
            assert wloss(y_test, y_preds).numpy() == pytest.approx(0.968, 0.01)
        if architecture == "tinho":
            assert wloss(y_test, y_preds).numpy() == pytest.approx(0.836, 0.01)

    @pytest.mark.parametrize(
        ("architecture", "dataset", "model_name"),
        (
            (
                "tinho",
                "plasticc",
                "model-GR-noZ-28341-1654269564-0.5.1.dev73+g70f85f8-LL0.836.tflite",
            ),
            (
                "tinho-quantized",
                "plasticc",
                "quantized-model-GR-noZ-28341-1654269564-0.5.1.dev73+g70f85f8-LL0.836.tflite",
            ),
        ),
    )
    def test_inference_GR_noZ_TFLITE(
        self, architecture, dataset, model_name, get_fixt_UGRIZY_wZ
    ):

        X_test, y_test, _ = get_fixt_UGRIZY_wZ
        X_test = X_test[:, :, 0:3:2]

        test_input = X_test

        test_ds = (
            tf.data.Dataset.from_tensor_slices((test_input, y_test))
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        y_test_ds = (
            tf.data.Dataset.from_tensor_slices(y_test)
            .batch(BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

        if LOCAL_DEBUG is not None:
            print("LOCAL_DEBUG set, reducing dataset size...")
            test_ds = test_ds.take(300)
            y_test_ds = y_test_ds.take(300)

        y_test = np.concatenate([y for y in y_test_ds], axis=0)
        x_test = np.concatenate([x for x, y in test_ds], axis=0)

        model_path = f"{asnwd}/astronet/tinho/models/{dataset}/{model_name}"
        model = LiteModel.from_file(model_path=model_path)

        wloss = WeightedLogLoss()
        y_preds = model.predict(x_test)

        if architecture == "tinho":
            loss = wloss(y_test, y_preds).numpy()
            log.info(f"LOSS tinho: {loss:.3f}")
            assert loss == pytest.approx(0.836, 0.001)
        if architecture == "tinho-quantized":
            loss = wloss(y_test, y_preds).numpy()
            log.info(f"LOSS tinho-quantized: {loss:.3f}")
            assert loss == pytest.approx(0.834, 0.001)
