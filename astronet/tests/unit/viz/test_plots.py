import json
import random as python_random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.metrics import WeightedLogLoss
from astronet.tests.conftest import ISA
from astronet.utils import get_encoding
from astronet.viz.visualise_results import (
    plot_acc_history,
    plot_confusion_matrix,
    plot_loss_history,
    plot_multiPR,
    plot_multiROC,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)

if ISA == "arm64":
    hashlib = (f"{Path(__file__).absolute().parent}/baseline/hashlib.json",)
else:
    hashlib = (f"{Path(__file__).absolute().parent}/baseline/m1-hashlib.json",)


@pytest.mark.parametrize(
    ("architecture", "dataset", "model_name"),
    (
        ("atx", "plasticc", "9887359-1641295475-0.1.dev943+gc9bafac.d20220104"),
        ("t2", "plasticc", "1619624444-0.1.dev765+g7c90cbb.d20210428"),
        ("t2", "plasticc", "31367-1654360237-0.5.1.dev78+g702e399.d20220604.png"),
    ),
)
class TestPlots:
    """A class with common parameters, `architecture`, `dataset` and `model_name`."""

    def compute_scores(self, architecture, dataset, model_name, fixt):

        X_test, y_test, Z_test, inputs = fixt

        results_filename = (
            f"{asnwd}/astronet/{architecture}/models/{dataset}/results_with_z.json"
        )

        with open(results_filename) as f:
            events = json.load(f)
            if model_name is not None:
                # Get params for model chosen with cli args
                event = next(
                    item
                    for item in events["training_result"]
                    if item["name"] == model_name
                )
            else:
                event = min(
                    events["training_result"],
                    key=lambda ev: ev["model_evaluate_on_test_loss"],
                )

        model = keras.models.load_model(
            f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

        dataform = "testset"
        encoding, class_encoding, class_names = get_encoding(dataset, dataform=dataform)

        y_preds = model.predict([X_test, Z_test])

        return (
            event,
            encoding,
            class_names,
            model,
            y_preds,
        )

    def test_params(self, architecture, dataset, model_name):
        print("\ntest_one", architecture, dataset, model_name)

    def test_fixtures(self, architecture, dataset, model_name, fixt):
        print("\ntest_one", architecture, dataset, model_name, fixt)

    @pytest.mark.mpl_image_compare(
        hash_library=f"{Path(__file__).absolute().parent}/baseline/hashlib.json",
    )
    def test_succeeds(self, architecture, dataset, model_name):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2, 3])
        print("\ntest_one", architecture, dataset, model_name)
        return fig

    @pytest.mark.mpl_image_compare(
        hash_library=f"{Path(__file__).absolute().parent}/baseline/hashlib.json",
    )
    def test_acc_history(self, architecture, dataset, model_name, fixt):

        X_test, y_test, Z_test, inputs = fixt

        (
            event,
            encoding,
            class_names,
            model,
            y_preds,
        ) = self.compute_scores(architecture, dataset, model_name, fixt)

        fig = plot_acc_history(
            architecture,
            dataset,
            model_name,
            event,
            save=False,
        )
        return fig

    @pytest.mark.mpl_image_compare(
        hash_library=f"{Path(__file__).absolute().parent}/baseline/hashlib.json",
    )
    def test_loss_history(self, architecture, dataset, model_name, fixt):

        X_test, y_test, Z_test, inputs = fixt

        (
            event,
            encoding,
            class_names,
            model,
            y_preds,
        ) = self.compute_scores(architecture, dataset, model_name, fixt)

        fig = plot_loss_history(
            architecture,
            dataset,
            model_name,
            event,
            save=False,
        )
        return fig

    @pytest.mark.mpl_image_compare(
        hash_library=f"{Path(__file__).absolute().parent}/baseline/hashlib.json",
    )
    def test_confusion_matrix(self, architecture, dataset, model_name, fixt):

        cmap = sns.light_palette("Navy", as_cmap=True)

        X_test, y_test, Z_test, inputs = fixt

        (
            event,
            encoding,
            class_names,
            model,
            y_preds,
        ) = self.compute_scores(architecture, dataset, model_name, fixt)

        fig = plot_confusion_matrix(
            architecture,
            dataset,
            model_name,
            y_test,
            y_preds,
            encoding,
            class_names,  # enc.categories_[0]
            save=False,
            cmap=cmap,
        )
        return fig

    @pytest.mark.mpl_image_compare(
        hash_library=f"{Path(__file__).absolute().parent}/baseline/hashlib.json",
    )
    def test_multiROC(self, architecture, dataset, model_name, fixt):

        X_test, y_test, Z_test, inputs = fixt

        (
            event,
            encoding,
            class_names,
            model,
            y_preds,
        ) = self.compute_scores(architecture, dataset, model_name, fixt)

        fig = plot_multiROC(
            architecture,
            dataset,
            model_name,
            model,
            inputs,
            y_test,
            class_names,
            save=False,
        )
        return fig

    @pytest.mark.mpl_image_compare(
        hash_library=f"{Path(__file__).absolute().parent}/baseline/hashlib.json",
    )
    def test_multiPR(self, architecture, dataset, model_name, fixt):

        X_test, y_test, Z_test, inputs = fixt

        (
            event,
            encoding,
            class_names,
            model,
            y_preds,
        ) = self.compute_scores(architecture, dataset, model_name, fixt)

        fig = plot_multiPR(
            architecture,
            dataset,
            model_name,
            model,
            inputs,
            y_test,
            class_names,
            save=False,
        )
        return fig
