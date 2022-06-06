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


# @pytest.mark.skipif(
#     os.getenv("CI") is not None, reason="Requires large data files. Run locally."
# )
# @pytest.mark.parametrize(
#     ("architecture", "dataset", "model_name"),
#     (
#         ("atx", "plasticc", "9887359-1641295475-0.1.dev943+gc9bafac.d20220104"),
#         ("t2", "plasticc", "1619624444-0.1.dev765+g7c90cbb.d20210428"),
#         ("tinho", "plasticc", "1619624444-0.1.dev765+g7c90cbb.d20210428"),
#     ),
# )
# class TestLiteInference:
#     """Test inference scores of the best performing models to date. This class is also a proxy for
#     inference timing of each model"""
#     def __init__(self, fixt):
#         pass

#     def test_inference_with_z(self, architecture, dataset, model_name, fixt):

#         X_test, y_test, Z_test, inputs = fixt

#         model = keras.models.load_model(
#             f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}",
#             custom_objects={"WeightedLogLoss": WeightedLogLoss()},
#             compile=False,
#         )

#         wloss = WeightedLogLoss()
#         y_preds = model.predict([X_test, Z_test])
#         # if architecture == "atx":
#         #     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.739, 0.01)
#         # if architecture == "t2":
#         #     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)
#         # if architecture == "t2":
#         #     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)

#         match architecture:
#                 case "atx":
#                     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.739, 0.01)
#                 case "t2":
#                     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)
#                 case "tinho":
#                     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)
#                 case _:
#                     return

#     def test_inference_no_z(self, architecture, dataset, model_name, fixt):

#         X_test, y_test, Z_test, inputs = fixt

#         model = keras.models.load_model(
#             f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}",
#             custom_objects={"WeightedLogLoss": WeightedLogLoss()},
#             compile=False,
#         )

#         wloss = WeightedLogLoss()
#         y_preds = model.predict(X_test)
#         # if architecture == "atx":
#         #     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.739, 0.01)
#         # if architecture == "t2":
#         #     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)
#         # if architecture == "t2":
#         #     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)

#         match architecture:
#                 case "atx":
#                     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.739, 0.01)
#                 case "t2":
#                     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)
#                 case "tinho":
#                     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)
#                 case _:
#                     return

#     def test_inference_ztf(self, architecture, dataset, model_name, fixt):

#         X_test, y_test, Z_test, inputs = fixt

#         model = keras.models.load_model(
#             f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}",
#             custom_objects={"WeightedLogLoss": WeightedLogLoss()},
#             compile=False,
#         )

#         wloss = WeightedLogLoss()
#         y_preds = model.predict([X_test, Z_test])

#         match architecture:
#                 case "atx":
#                     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.739, 0.01)
#                 case "t2":
#                     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)
#                 case "tinho":
#                     assert wloss(y_test, y_preds).numpy() == pytest.approx(0.507, 0.01)
#                 case _:
#                     return
