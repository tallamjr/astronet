import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras.backend as K

from astronet.t2.model import T2Model


@pytest.mark.parametrize(
    (
        "droprate",
        "embed_dim",
        "ff_dim",
        "num_heads",
        "num_layers",
        "num_aux_feats",
        "answer",
    ),
    (
        (0.1, 32, 128, 16, 1, 0, 13390),
        (0.1, 32, 128, 16, 1, 2, 13390),
    ),
)
def test_num_parameters(
    droprate, embed_dim, ff_dim, num_heads, num_layers, num_aux_feats, answer
):

    # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
    num_filters = embed_dim

    input_shape = (None, 100, 6)
    model = T2Model(
        input_shape,
        embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_filters=num_filters,
        num_classes=14,
        num_layers=num_layers,
        droprate=droprate,
        num_aux_feats=num_aux_feats,
        add_aux_feats_to="L",
        # Either add features to M dimension or L dimension. Adding to L allows for
        # visualisation of CAMs relating to redshift since we would have a CAM of (L + Z) x c
        # fc_neurons=fc_neurons,
    )
    inputs = tf.keras.Input(shape=[100, 6])
    model(inputs)
    assert np.sum([K.count_params(p) for p in model.trainable_weights]) == answer


@pytest.mark.xfail(reason="model_profiler version issue. To be investigated")
@pytest.mark.parametrize(
    (
        "droprate",
        "embed_dim",
        "ff_dim",
        "num_heads",
        "num_layers",
        "num_aux_feats",
        "BATCH_SIZE",
    ),
    (
        (0.1, 32, 128, 16, 1, 0, 2048),
        # (0.1, 32, 128, 16, 1, 0, 2048),
    ),
)
def test_model_size(
    droprate,
    embed_dim,
    ff_dim,
    num_heads,
    num_layers,
    num_aux_feats,
    BATCH_SIZE,
):

    from model_profiler import model_profiler

    # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
    num_filters = embed_dim

    input_shape = (BATCH_SIZE, 100, 6)

    model = T2Model(
        input_dim=input_shape,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_filters=num_filters,
        num_classes=14,
        num_layers=num_layers,
        droprate=droprate,
        num_aux_feats=num_aux_feats,
        add_aux_feats_to="L",
        # Either add features to M dimension or L dimension. Adding to L allows for
        # visualisation of CAMs relating to redshift since we would have a CAM of (L + Z) x c
        # fc_neurons=fc_neurons,
    )

    inputs = tf.keras.Input(shape=[100, 6])

    model.build_graph(input_shape)
    model.build(input_shape)

    model(inputs)

    # profile = model_profiler(model, BATCH_SIZE)
    # print(profile)
