import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from astronet.t2.transformer import (
    ConvEmbedding,
    PositionalEncoding,
    RelativePositionEmbedding,
    TransformerBlock,
)


def build_model(
    input_shapes,
    input_dim,
    embed_dim,
    num_heads,
    ff_dim,
    num_filters,
    num_classes,
    num_layers,
    droprate,
    num_aux_feats=0,
    add_aux_feats_to="L",
    **kwargs,
):

    if isinstance(input_shapes, tuple):  # A list would imply there is multiple inputs
        # Code lifted from example:
        # https://github.com/tensorflow/tensorflow/issues/29132#issuecomment-504679288
        input_shape_nobatch = input_shapes[1:]
        # build(input_shapes)
        inputs = keras.Input(shape=input_shape_nobatch)
    else:
        input_shape_nobatch = input_shapes[0][1:]
        Z_input_shape_nobatch = input_shapes[1][1:]
        inputs = [
            tf.keras.Input(shape=input_shape_nobatch),
            tf.keras.Input(shape=Z_input_shape_nobatch),
        ]

    ##############################################################
    if add_aux_feats_to == "L":
        sequence_length = input_dim[1] + num_aux_feats
    else:
        sequence_length = input_dim[
            1
        ]  # input_dim.shape = (batch_size, input_seq_len, d_model)

    # embedding = ConvEmbedding(num_filters=num_filters, input_shape=input_dim)

    # # <-- Additional layers when adding Z features here -->

    # pos_encoding = PositionalEncoding(max_steps=sequence_length, max_dims=embed_dim)

    # encoder = [
    #     TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
    # ]

    # pooling = layers.GlobalAveragePooling1D()
    # dropout1 = layers.Dropout(droprate)

    # # self.fc             = layers.Dense(self.embed_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    # # self.dropout2       = layers.Dropout(self.droprate)

    # classifier = layers.Dense(num_classes, activation="softmax")

    ##############################################################

    # If not a list then inputs are of type tensor: tf.is_tensor(inputs) == True
    if tf.is_tensor(inputs):
        x = ConvEmbedding(num_filters=num_filters, input_shape=input_dim)(inputs)
        x = PositionalEncoding(max_steps=sequence_length, max_dims=embed_dim)(x)

        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x, training)

        x = layers.GlobalAveragePooling1D()(x)

        if training:
            x = layers.Dropout(droprate)(x, training=training)

        classifier = layers.Dense(num_classes, activation="softmax")(x)

    # if (isinstance(inputs, list)) and (self.add_aux_feats_to == "M"):
    # Else this implies input is a list; a list of tensors, i.e. multiple inputs
    else:
        # X in L x M
        x = inputs[0]
        # Additional Z features
        z = inputs[1]
        # >>> z.shape
        # TensorShape([None, 2])
        if add_aux_feats_to == "M":
            z = tf.tile(z, [1, 100])
            # >>> z.shape
            # TensorShape([None, 200])
            z = tf.keras.layers.Reshape([100, 2])(z)
            # >>> z.shape
            # TensorShape([None, 100, 2])
            x = tf.keras.layers.Concatenate(axis=2)([x, z])
            # >>> x.shape
            # TensorShape([None, 100, 8)])
        else:  # Else self.add_aux_feats_to == 'L'
            z = tf.tile(z, [1, 6])
            # >>> z.shape
            # TensorShape([None, 12])
            z = tf.keras.layers.Reshape([2, 6])(z)
            # >>> z.shape
            # TensorShape([None, 2, 6])
            x = tf.keras.layers.Concatenate(axis=1)([x, z])
            # >>> x.shape
            # TensorShape([None, 102, 6)])

        # Transforms X in L x (M + Z) -> X in L x d if self.add_aux_feats_to == "M" or
        # transforms X in (L + 2) x M -> X in L x d if self.add_aux_feats_to == "L"

        x = ConvEmbedding(num_filters=num_filters, input_shape=input_dim)(x)
        x = PositionalEncoding(max_steps=sequence_length, max_dims=embed_dim)(
            x
        )  # X <- X + P, where X in L x d

        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x, training)

        x = layers.GlobalAveragePooling1D()(x)
        if training:
            x = layers.Dropout(droprate)(x, training=training)

        classifier = layers.Dense(num_classes, activation="softmax")(x)

    return classifier
