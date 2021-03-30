import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from astronet.t2.attention import MultiHeadSelfAttention


class ConvEmbedding(layers.Layer):
    def __init__(self, num_filters, **kwargs):
        super(ConvEmbedding, self).__init__(**kwargs)
        self.conv1d = layers.Conv1D(
            filters=num_filters, kernel_size=1, activation="relu"
        )

    def call(self, inputs):
        embedding = self.conv1d(inputs)

        return embedding


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
        super(PositionalEncoding, self).__init__(dtype=dtype, **kwargs)
        if max_dims % 2 == 1:
            max_dims += 1  # max_dims must be even
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / max_dims)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))

    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]


class RelativePositionEmbedding(tf.keras.layers.Layer):
    """Creates a positional embedding.

    This layer calculates the position encoding as a mix of sine and cosine
    functions with geometrically increasing wavelengths. Defined and formulized in
    "Attention is All You Need", section 3.5.
    (https://arxiv.org/abs/1706.03762).

    Arguments:
    hidden_size: Size of the hidden layer.
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position.
    """

    def __init__(self,
               hidden_size,
               min_timescale=1.0,
               max_timescale=1.0e4,
               **kwargs):
        # We need to have a default dtype of float32, since the inputs (which Keras
        # usually uses to infer the dtype) will always be int32.
        # We compute the positional encoding in float32 even if the model uses
        # float16, as many of the ops used, like log and exp, are numerically
        # unstable in float16.
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"

        super(RelativePositionEmbedding, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._min_timescale = min_timescale
        self._max_timescale = max_timescale

    def call(self, inputs, length=None):
        """Implements call() for the layer.

        Args:
          inputs: An tensor whose second dimension will be used as `length`. If
            `None`, the other `length` argument must be specified.
          length: An optional integer specifying the number of positions. If both
            `inputs` and `length` are spcified, `length` must be equal to the second
            dimension of `inputs`.

        Returns:
          A tensor in shape of [length, hidden_size].
        """
        shape = tf.shape(inputs)
        length = shape[1]
        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = self._hidden_size // 2
        min_timescale, max_timescale = self._min_timescale, self._max_timescale
        log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
            tf.cast(num_timescales, tf.float32) - 1
        )
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment
        )
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        position_embeddings = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        return inputs + position_embeddings


class TransformerBlock(layers.Layer):
    # TODO: Update docstrings
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):

        # Sublayer 1
        attn_output = self.att(inputs)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Residual connection, (batch_size, input_seq_len, d_model)

        # Sublayer 2
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection, # (batch_size, input_seq_len, d_model)

        return out2  # (batch_size, input_seq_len, d_model)


class AdditionalFeatures(layers.Layer):
    def __init__(self, **kwargs):
        super(AdditionalFeatures, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs[0]
        z = inputs[1]
        z = tf.broadcast_to(z, shape=x.shape)
        x = tf.keras.layers.Concatenate(axis=1)([z, x])

        return x
