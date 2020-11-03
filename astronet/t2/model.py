from tensorflow import keras
from tensorflow.keras import layers

from astronet.t2.transformer import ConvEmbedding, TransformerBlock


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

        self.embedding      = ConvEmbedding(num_filters=self.num_filters, input_shape=input_dim[1:])
        self.encoder        = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        self.pooling        = layers.GlobalAveragePooling1D()
        self.dropout1       = layers.Dropout(0.1)
        self.fc             = layers.Dense(20, activation="relu")
        self.dropout2       = layers.Dropout(0.1)
        self.classifier     = layers.Dense(self.num_classes, activation="softmax")

    def call(self, inputs, training=False):

        embedding   = self.embedding(inputs)
        encoder     = self.encoder(embedding)
        pooling     = self.pooling(encoder)
        dropout1    = self.dropout1(pooling)
        fc          = self.fc(dropout1)
        dropout2    = self.dropout2(fc)
        classifier  = self.classifier(dropout2)

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
