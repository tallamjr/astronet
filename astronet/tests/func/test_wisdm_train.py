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

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.backend import clear_session

from astronet.t2.model import T2Model
from astronet.tests.conftest import SKIP_IF_M1
from astronet.utils import astronet_logger, load_dataset

log = astronet_logger(__file__)
log.info("=" * shutil.get_terminal_size((80, 20))[0])
log.info(f"File Path: {Path(__file__).absolute()}")
log.info(f"Parent of Directory Path: {Path().absolute().parent}")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Requires large datafile")
@SKIP_IF_M1
def test_training_pipeline_wisdm_2010():
    clear_session()

    # Load WISDM-2010
    X_train, y_train, X_test, y_test, loss = load_dataset("wisdm_2010")

    num_classes = y_train.shape[1]

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    BATCH_SIZE = 32
    EPOCHS = 2

    print(type(X_train))

    embed_dim = 32  # --> Embedding size for each token
    num_heads = 4  # --> Number of attention heads
    ff_dim = 32  # --> Hidden layer size in feed forward network inside transformer

    # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
    num_filters = embed_dim

    num_layers = 1  # --> N x repeated transformer blocks
    droprate = 0.1  # --> Rate of neurons to drop

    (
        _,
        timesteps,
        num_features,
    ) = X_train.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)
    input_shape = (BATCH_SIZE, timesteps, num_features)
    print(input_shape)

    model = T2Model(
        input_dim=input_shape,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_filters=num_filters,
        num_classes=num_classes,
        num_layers=num_layers,
        droprate=droprate,
    )

    model.compile(loss=loss, optimizer="adam", metrics=["acc"])

    _ = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
    )

    model.build_graph(input_shape)

    print(model.summary())

    print(model.evaluate(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    expected_output = [0.44523268938064575, 0.7262773513793945]
    assert accuracy == pytest.approx(expected_output[1], 0.1)


# @pytest.mark.skipif(os.getenv("CI") is not None, reason="Requires large datafile")
# def test_training_pipeline_plasticc():
#     clear_session()

#     # Load WISDM-2010
#     X_train, y_train, X_test, y_test, wloss = load_dataset("plasticc", snonly=True)

#     num_classes = y_train.shape[1]

#     print(X_train.shape, y_train.shape)
#     print(X_test.shape, y_test.shape)

#     BATCH_SIZE = 32
#     EPOCHS = 2

#     print(type(X_train))

#     embed_dim = 32  # --> Embedding size for each token
#     num_heads = 4  # --> Number of attention heads
#     ff_dim = 32  # --> Hidden layer size in feed forward network inside transformer

#     # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
#     num_filters = embed_dim

#     num_layers = 1  # --> N x repeated transformer blocks
#     droprate = 0.1  # --> Rate of neurons to drop

#     (
#         _,
#         timesteps,
#         num_features,
#     ) = X_train.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)
#     input_shape = (BATCH_SIZE, timesteps, num_features)
#     print(input_shape)

#     model = T2Model(
#         input_dim=input_shape,
#         embed_dim=embed_dim,
#         num_heads=num_heads,
#         ff_dim=ff_dim,
#         num_filters=num_filters,
#         num_classes=num_classes,
#         num_layers=num_layers,
#         droprate=droprate,
#     )

#     # wloss = WeightedLogLoss()
#     # wloss = custom_log_loss

#     model.compile(
#         loss=wloss,
#         optimizer="adam",
#         metrics=["acc"],
#         run_eagerly=True,
#     )

#     _ = model.fit(
#         X_train,
#         y_train,
#         batch_size=BATCH_SIZE,
#         epochs=EPOCHS,
#         validation_data=(X_test, y_test),
#     )

#     model.build_graph(input_shape)

#     print(model.summary())

#     print(model.evaluate(X_test, y_test))

#     loss, accuracy = model.evaluate(X_test, y_test)
#     expected_output = [0.44523268938064575, 0.6452905535697937]
#     assert accuracy == pytest.approx(expected_output[1], 0.1)


# @pytest.mark.skipif(os.getenv("CI") is not None, reason="Requires large datafile")
# def test_training_pipeline_full_plasticc():
#     clear_session()

#     # Load WISDM-2010
#     X_train, y_train, X_test, y_test, wloss = load_dataset("plasticc")

#     num_classes = y_train.shape[1]

#     print(X_train.shape, y_train.shape)
#     print(X_test.shape, y_test.shape)

#     BATCH_SIZE = 32
#     EPOCHS = 2

#     print(type(X_train))

#     embed_dim = 32  # --> Embedding size for each token
#     num_heads = 4  # --> Number of attention heads
#     ff_dim = 32  # --> Hidden layer size in feed forward network inside transformer

#     # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
#     num_filters = embed_dim

#     num_layers = 1  # --> N x repeated transformer blocks
#     droprate = 0.1  # --> Rate of neurons to drop

#     (
#         _,
#         timesteps,
#         num_features,
#     ) = X_train.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)
#     input_shape = (BATCH_SIZE, timesteps, num_features)
#     print(input_shape)

#     model = T2Model(
#         input_dim=input_shape,
#         embed_dim=embed_dim,
#         num_heads=num_heads,
#         ff_dim=ff_dim,
#         num_filters=num_filters,
#         num_classes=num_classes,
#         num_layers=num_layers,
#         droprate=droprate,
#     )

#     # wloss = WeightedLogLoss()
#     # wloss = custom_log_loss

#     model.compile(
#         loss=wloss,
#         optimizer="adam",
#         metrics=["acc"],
#         run_eagerly=True,
#     )

#     _ = model.fit(
#         X_train,
#         y_train,
#         batch_size=BATCH_SIZE,
#         epochs=EPOCHS,
#         validation_data=(X_test, y_test),
#     )

#     model.build_graph(input_shape)

#     print(model.summary())

#     print(model.evaluate(X_test, y_test))

#     loss, accuracy = model.evaluate(X_test, y_test)
#     expected_output = [0.44523268938064575, 0.6452905535697937]
#     assert accuracy == pytest.approx(expected_output[1], 0.1)
