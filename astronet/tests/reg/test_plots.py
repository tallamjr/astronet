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

import json
import os
import random as python_random
from pathlib import Path

import matplotlib as mpl
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

mpl.rcParams["savefig.format"] = "pdf"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(RANDOM_SEED)

hashlib = f"{Path(__file__).absolute().parent}/baseline/{ISA}-hashlib.json"


@pytest.mark.skipif(
    os.getenv("CI") is not None, reason="Requires large data files. Run locally."
)
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
class TestPlots:
    """A class with common parameters, `architecture`, `dataset` and `model_name`."""

    def compute_scores(self, architecture, dataset, model_name, get_fixt_UGRIZY_wZ):

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

        test_ds, y_test_ds, test_inputs = get_fixt_UGRIZY_wZ

        model = keras.models.load_model(
            f"{asnwd}/astronet/{architecture}/models/{dataset}/model-{model_name}",
            custom_objects={"WeightedLogLoss": WeightedLogLoss()},
            compile=False,
        )

        y_preds = model.predict(test_inputs)

        dataform = "testset"
        encoding, class_encoding, class_names = get_encoding(dataset, dataform=dataform)

        return (
            event,
            encoding,
            class_names,
            model,
            y_preds,
        )

    def test_params(self, architecture, dataset, model_name):
        print("\ntest_one", architecture, dataset, model_name)

    def test_fixtures(self, architecture, dataset, model_name, get_fixt_UGRIZY_wZ):
        print("\ntest_one", architecture, dataset, model_name, get_fixt_UGRIZY_wZ)

    def test_succeeds(self, architecture, dataset, model_name):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot([1, 2, 3])
        print("\ntest_one", architecture, dataset, model_name)
        return fig

    @pytest.mark.mpl_image_compare(
        # hash_library=hashlib,
    )
    def test_acc_history(self, architecture, dataset, model_name, get_fixt_UGRIZY_wZ):

        test_ds, y_test_ds, test_inputs = get_fixt_UGRIZY_wZ

        (
            event,
            encoding,
            class_names,
            model,
            y_preds,
        ) = self.compute_scores(architecture, dataset, model_name, get_fixt_UGRIZY_wZ)

        fig = plot_acc_history(
            architecture,
            dataset,
            model_name,
            event,
            save=False,
        )
        return fig

    @pytest.mark.mpl_image_compare(
        # hash_library=hashlib,
    )
    def test_loss_history(self, architecture, dataset, model_name, get_fixt_UGRIZY_wZ):

        test_ds, y_test_ds, test_inputs = get_fixt_UGRIZY_wZ

        (
            event,
            encoding,
            class_names,
            model,
            y_preds,
        ) = self.compute_scores(architecture, dataset, model_name, get_fixt_UGRIZY_wZ)

        fig = plot_loss_history(
            architecture,
            dataset,
            model_name,
            event,
            save=False,
        )
        return fig

    @pytest.mark.mpl_image_compare(
        # hash_library=hashlib,
    )
    def test_confusion_matrix(
        self, architecture, dataset, model_name, get_fixt_UGRIZY_wZ
    ):

        test_ds, y_test_ds, test_inputs = get_fixt_UGRIZY_wZ
        y_test = np.concatenate([y for y in y_test_ds], axis=0)

        (
            event,
            encoding,
            class_names,
            model,
            y_preds,
        ) = self.compute_scores(architecture, dataset, model_name, get_fixt_UGRIZY_wZ)

        cmap = sns.light_palette("Navy", as_cmap=True)
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
        # hash_library=hashlib,
    )
    def test_multiROC(self, architecture, dataset, model_name, get_fixt_UGRIZY_wZ):

        test_ds, y_test_ds, test_inputs = get_fixt_UGRIZY_wZ
        y_test = np.concatenate([y for y in y_test_ds], axis=0)

        (
            event,
            encoding,
            class_names,
            model,
            y_preds,
        ) = self.compute_scores(architecture, dataset, model_name, get_fixt_UGRIZY_wZ)

        fig = plot_multiROC(
            architecture,
            dataset,
            model_name,
            model,
            y_test,
            y_preds,
            class_names,
            save=False,
        )
        return fig

    @pytest.mark.mpl_image_compare(
        # hash_library=hashlib,
    )
    def test_multiPR(self, architecture, dataset, model_name, get_fixt_UGRIZY_wZ):

        test_ds, y_test_ds, test_inputs = get_fixt_UGRIZY_wZ
        y_test = np.concatenate([y for y in y_test_ds], axis=0)

        (
            event,
            encoding,
            class_names,
            model,
            y_preds,
        ) = self.compute_scores(architecture, dataset, model_name, get_fixt_UGRIZY_wZ)

        fig = plot_multiPR(
            architecture,
            dataset,
            model_name,
            model,
            y_test,
            y_preds,
            class_names,
            save=False,
        )
        return fig
