import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from astronet.t2.attention import MultiHeadSelfAttention

# embed_dim = 32    --> Embedding size for each token
# num_heads = 4     --> Number of attention heads
# ff_dim = 32       --> Hidden layer size in feed forward network inside transformer


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class ConvEmbedding(layers.Layer):
    def __init__(self, **kwargs):
        super(ConvEmbedding, self).__init__(**kwargs)
        self.conv1d = layers.Conv1D(32, kernel_size=1, activation='relu')

    def call(self, inputs):
        embedding = self.conv1d(inputs)

        return embedding
