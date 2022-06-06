import os
import random as python_random

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.metrics import WeightedLogLoss

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
            ("atx", "plasticc", "9887359-1641295475-0.1.dev943+gc9bafac.d20220104"),
            ("t2", "plasticc", "1619624444-0.1.dev765+g7c90cbb.d20210428"),
        ),
    )
    def test_inference_with_z_numpy(
        self, architecture, dataset, model_name, fixt_numpy
    ):

        # Previous models were trained using numpy data as the inputs, newer models leverage
        # tf.data.Dataset instead for faster inference. This is a legacy requirment.
        # Fix ValueError of shape mismatch.
        X_test, y_test, Z_test, inputs = fixt_numpy

        model = keras.models.load_model(
            f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

        wloss = WeightedLogLoss()
        y_preds = model.predict(inputs)

        if architecture == "atx":
            assert wloss(y_test, y_preds).numpy() == pytest.approx(0.739, 0.01)
        if architecture == "t2":
            assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)

    @pytest.mark.parametrize(
        ("architecture", "dataset", "model_name"),
        (("tinho", "plasticc", "31367-1654360237-0.5.1.dev78+g702e399.d20220604"),),
    )
    def test_inference_with_z_tfdata(self, architecture, dataset, model_name, fixt):

        test_ds, y_test_ds = fixt
        y_test = np.concatenate([y for y in y_test_ds], axis=0)

        model = keras.models.load_model(
            f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

        wloss = WeightedLogLoss()
        y_preds = model.predict(test_ds)

        if architecture == "tinho":
            assert wloss(y_test, y_preds).numpy() == pytest.approx(0.450, 0.01)

    # def test_inference_no_z(self, architecture, dataset, model_name, fixtnoz):

    #     test_ds, y_test_ds = fixtnoz
    #     y_test = np.concatenate([y for y in y_test_ds], axis=0)

    #     model = keras.models.load_model(
    #         f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}",
    #         custom_objects={"WeightedLogLoss": WeightedLogLoss()},
    #         compile=False,
    #     )

    #     wloss = WeightedLogLoss()
    #     y_preds = model.predict(test_ds)

    #     if architecture == "atx":
    #         assert wloss(y_test, y_preds).numpy() == pytest.approx(0.739, 0.01)
    #     if architecture == "t2":
    #         assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)
    #     if architecture == "t2":
    #         assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)

    # def test_inference_ztf(self, architecture, dataset, model_name, fixtztf):

    #     test_ds, y_test_ds = fixtztf
    #     y_test = np.concatenate([y for y in y_test_ds], axis=0)

    #     model = keras.models.load_model(
    #         f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}",
    #         custom_objects={"WeightedLogLoss": WeightedLogLoss()},
    #         compile=False,
    #     )

    #     wloss = WeightedLogLoss()
    #     y_preds = model.predict(test_ds)

    #     if architecture == "atx":
    #         assert wloss(y_test, y_preds).numpy() == pytest.approx(0.739, 0.01)
    #     if architecture == "t2":
    #         assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)
    #     if architecture == "t2":
    #         assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)
