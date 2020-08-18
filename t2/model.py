import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from astronet.t2.transformer import ConvEmbedding, EncoderBlock, DecoderBlock, ClassifyBlock


class T2Model(keras.Model):
    """Time-Transformer with Multi-headed."""
    def __init__(self):
        super(T2Model, self).__init__()
        self.embedding  = ConvEmbedding()
        self.encoder    = EcoderBlock()
        self.decoder    = DecoderBlock()
        self.classifier = ClassifyBlock()

    def call(self, inputs):
        model = keras.Sequential()

        # model.add(layers.Dense(units=128))

        # Old API
        # model.add(layers.Convolution1D(filters=128, kernel_size=16, activation='relu'))
        # New Keras API
        # model.add(layers.Conv1D(filters=128, kernel_size=16, activation='relu', input_shape=input_shape[1:]))

        # Inspired by this line: Note that a TimeDistributed(Dense(n)) layer is equivalent to a Conv1D(n, filter_size=1) layer.
        # Moving kernel_size from 16 to 1
        # Each time-step becomes a 32-vector, where one can think of a window of 200 time-steps being equivulent
        # to a sentence in NLP land
        model.add(layers.Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=input_shape[1:]))

        model.add(TransformerBlock(embed_dim, num_heads, ff_dim))
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(20, activation="relu"))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(6, activation="softmax"))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
