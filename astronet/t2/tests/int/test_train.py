import pytest
import numpy as np
import tensorflow as tf

from astronet.t2.model import T2Model
from astronet.t2.utils import t2_logger, load_wisdm_2010
from astronet.t2.preprocess import one_hot_encode

from pathlib import Path

log = t2_logger(__file__)
log.info("_________________________________")
log.info(f"File      Path: {Path(__file__).absolute()}")
log.info(f"Parent of Directory Path: {Path().absolute().parent}")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def test_training_pipeline_wisdm_2010():

    # Load WISDM-2010
    X_train, y_train, X_val, y_val, X_test, y_test = load_wisdm_2010()
    # One hot encode y
    enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)
    num_classes = y_train.shape[1]

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    BATCH_SIZE = 32
    EPOCHS = 2

    print(type(X_train))

    embed_dim = 32  # --> Embedding size for each token
    num_heads = 4  # --> Number of attention heads
    ff_dim = 32  # --> Hidden layer size in feed forward network inside transformer

    # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
    num_filters = embed_dim

    input_shape = X_train.shape
    print(input_shape[1:])  # (TIMESTEPS, num_features)

    model = T2Model(
        input_dim=input_shape,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_filters=num_filters,
        num_classes=num_classes,
    )

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["acc"]
    )

    _ = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
    )

    model.build_graph(input_shape)

    print(model.summary())

    print(model.evaluate(X_test, y_test))

    result = model.evaluate(X_test, y_test)
    expected_output = [0.44523268938064575, 0.832602322101593]
    assert result[1] == pytest.approx(expected_output[1], 0.1)
