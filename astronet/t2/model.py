import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from astronet.t2.transformer import ConvEmbedding, PositionalEncoding, TransformerBlock


class T2Model(keras.Model):
    # TODO: Update docstrings
    """Time-Transformer with Multi-headed.
    embed_dim --> Embedding size for each token
    num_heads --> Number of attention heads
    ff_dim    --> Hidden layer size in feed forward network inside transformer
    """
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_filters, num_classes, **kwargs):
        super(T2Model, self).__init__()
        self.input_dim      = input_dim
        self.embed_dim      = embed_dim
        self.num_heads      = num_heads
        self.ff_dim         = ff_dim
        self.num_filters    = num_filters
        self.num_classes    = num_classes
        self.sequence_length = input_dim[1]   # input_dim.shape = (batch_size, input_seq_len, d_model)

        self.embedding      = ConvEmbedding(num_filters=self.num_filters, input_shape=input_dim)
        self.pos_encoding   = PositionalEncoding(max_steps=self.sequence_length, max_dims=self.embed_dim)
        self.encoder        = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        # TODO : Branch off here, outputs_2, with perhaps Dense(input_dim[1]), OR vis this layer since
        # output should be: (batch_size, input_seq_len, d_model), see:
        # https://github.com/cordeirojoao/ECG_Processing/blob/master/Ecg_keras_v9-Raphael.ipynb
        self.pooling        = layers.GlobalAveragePooling1D()
        self.dropout1       = layers.Dropout(0.1)
        self.fc             = layers.Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
        self.dropout2       = layers.Dropout(0.1)
        self.classifier     = layers.Dense(self.num_classes, activation="softmax")

    def call(self, inputs, training=None):

        x = self.embedding(inputs)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.pooling(x)
        if training:
            x = self.dropout1(x, training=training)
        x = self.fc(x)
        if training:
            x = self.dropout2(x, training=training)

        classifier = self.classifier(x)

        return classifier

    def build_graph(self, input_shape):
        # Code lifted from example:
        # https://github.com/tensorflow/tensorflow/issues/29132#issuecomment-504679288
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)
