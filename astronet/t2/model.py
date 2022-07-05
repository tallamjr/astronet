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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from astronet.t2.transformer import (
    ConvEmbedding,
    PositionalEncoding,
    RelativePositionEmbedding,
    TransformerBlock,
)


class T2Model(keras.Model):
    # TODO: Update docstrings
    """Time-Transformer with Multi-headed.
    embed_dim --> Embedding size for each token
    num_heads --> Number of attention heads
    ff_dim    --> Hidden layer size in feed forward network inside transformer
    """

    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        ff_dim,
        num_filters,
        num_classes,
        num_layers,
        droprate,
        num_aux_feats=0,
        add_aux_feats_to="M",
        **kwargs,
    ):
        super(T2Model, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.droprate = droprate
        self.num_aux_feats = num_aux_feats
        self.add_aux_feats_to = add_aux_feats_to

        self.num_classes = num_classes
        if self.add_aux_feats_to == "L":
            self.sequence_length = input_dim[1] + self.num_aux_feats
        else:
            self.sequence_length = input_dim[
                1
            ]  # input_dim.shape = (batch_size, input_seq_len, d_model)

        self.embedding = ConvEmbedding(
            num_filters=self.num_filters, input_shape=input_dim
        )

        # <-- Additional layers when adding Z features here -->

        self.pos_encoding = PositionalEncoding(
            max_steps=self.sequence_length, max_dims=self.embed_dim
        )

        self.encoder = [
            TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
            for _ in range(num_layers)
        ]

        self.pooling = layers.GlobalAveragePooling1D()
        self.dropout1 = layers.Dropout(self.droprate)

        # self.fc             = layers.Dense(self.embed_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
        # self.dropout2       = layers.Dropout(self.droprate)

        self.classifier = layers.Dense(self.num_classes, activation="softmax")

    def call(self, inputs, training=None):

        # If not a list then inputs are of type tensor: tf.is_tensor(inputs) == True
        if tf.is_tensor(inputs):
            x = self.embedding(inputs)
            x = self.pos_encoding(x)

            for layer in self.encoder:
                x = layer(x, training)

            x = self.pooling(x)
            if training:
                x = self.dropout1(x, training=training)

            # x = self.fc(x)
            # if training:
            #     x = self.dropout2(x, training=training)

            classifier = self.classifier(x)

        # if (isinstance(inputs, list)) and (self.add_aux_feats_to == "M"):
        # Else this implies input is a list; a list of tensors, i.e. multiple inputs
        else:
            if isinstance(inputs, dict):
                x = inputs["input_1"]
                z = inputs["input_2"]
            else:
                # X in L x M
                x = inputs[0]
                # Additional Z features
                z = inputs[1]
                # >>> z.shape
                # TensorShape([None, 2])
            if self.add_aux_feats_to == "M":
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
            x = self.embedding(x)

            x = self.pos_encoding(x)  # X <- X + P, where X in L x d

            for layer in self.encoder:
                x = layer(x, training)

            x = self.pooling(x)
            if training:
                x = self.dropout1(x, training=training)

            # Additional layers when adding Z features
            # x = tf.keras.layers.Concatenate(axis=1)([inputs[1], x])

            # x = self.fc(x)
            # if training:
            #     x = self.dropout2(x, training=training)

            classifier = self.classifier(x)

        return classifier

    def build_graph(self, input_shapes):
        if isinstance(
            input_shapes, tuple
        ):  # A list would imply there is multiple inputs
            # Code lifted from example:
            # https://github.com/tensorflow/tensorflow/issues/29132#issuecomment-504679288
            input_shape_nobatch = input_shapes[1:]
            # self.build(input_shapes)
            inputs = keras.Input(shape=input_shape_nobatch)
        else:
            input_shape_nobatch = input_shapes[0][1:]
            Z_input_shape_nobatch = input_shapes[1][1:]
            inputs = [
                tf.keras.Input(shape=input_shape_nobatch),
                tf.keras.Input(shape=Z_input_shape_nobatch),
            ]

        if not hasattr(self, "call"):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)
