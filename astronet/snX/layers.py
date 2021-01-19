import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    SeparableConv1D,
    Add,
    Dense,
    BatchNormalization,
    ReLU,
    MaxPool1D,
    GlobalAveragePooling1D,
)


class ConvBatchNormBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super(ConvBatchNormBlock, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv1d = Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="same",
            use_bias=False,
        )
        self.batchnorm = BatchNormalization()

    def call(self, inputs):
        x = inputs
        x = self.conv1d(x)
        x = self.batchnorm(x)
        return x


class SeparableConvBatchNormBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super(SeparableConvBatchNormBlock, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.sep_conv_1d = SeparableConv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="same",
            use_bias=False,
        )

        self.batchnorm = BatchNormalization()

    def call(self, inputs):
        x = inputs
        x = self.sep_conv_1d(x)
        x = self.batchnorm(x)
        return x


class EntryFlow(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EntryFlow, self).__init__(**kwargs)

        self.conv_batchnorm_1 = ConvBatchNormBlock(filters=32, kernel_size=3, strides=2)
        self.relu = ReLU()
        self.conv_batchnorm_2 = ConvBatchNormBlock(filters=64, kernel_size=3)
        # relu

        self.sep_conv_batchnorm_1 = SeparableConvBatchNormBlock(filters=128, kernel_size=3)
        # relu
        self.sep_conv_batchnorm_2 = SeparableConvBatchNormBlock(filters=128, kernel_size=3)
        # relu
        # maxpool

        self.conv_batchnorm_3 = ConvBatchNormBlock(filters=128, kernel_size=1, strides=2)

        # add
        # relu

        self.sep_conv_batchnorm_4 = SeparableConvBatchNormBlock(filters=256, kernel_size=3)
        # relu
        self.sep_conv_batchnorm_5 = SeparableConvBatchNormBlock(filters=256, kernel_size=3)
        # maxpool

        self.conv_batchnorm_4 = ConvBatchNormBlock(filters=256, kernel_size=1, strides=2)

        # add
        # relu

        self.sep_conv_batchnorm_6 = SeparableConvBatchNormBlock(filters=728, kernel_size=3)
        # relu
        self.sep_conv_batchnorm_7 = SeparableConvBatchNormBlock(filters=728, kernel_size=3)
        # maxpool
        self.maxpool = MaxPool1D(pool_size=3, strides=2, padding='same')

        self.conv_batchnorm_5 = ConvBatchNormBlock(filters=728, kernel_size=1, strides=2)
        # add

    def call(self, inputs):
        x = inputs
        x = self.conv_batchnorm_1(x)
        x = self.relu(x)
        x = self.conv_batchnorm_2(x)
        tensor = self.relu(x)

        x = self.sep_conv_batchnorm_1(tensor)
        x = self.relu(x)
        x = self.sep_conv_batchnorm_2(x)
        x = self.maxpool(x)

        tensor = self.conv_batchnorm_3(tensor)

        x = Add()([tensor, x])

        x = self.relu(x)
        x = self.sep_conv_batchnorm_4(x)
        x = self.relu(x)
        x = self.sep_conv_batchnorm_5(x)
        x = self.maxpool(x)

        tensor = self.conv_batchnorm_4(tensor)

        x = Add()([tensor, x])

        x = self.relu(x)
        x = self.sep_conv_batchnorm_6(x)
        x = self.relu(x)
        x = self.sep_conv_batchnorm_7(x)
        x = self.maxpool(x)

        tensor = self.conv_batchnorm_5(tensor)

        x = Add()([tensor, x])

        return x


class MiddleFlow(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MiddleFlow, self).__init__(**kwargs)

        self.relu = ReLU()
        self.sep_conv_batchnorm_mid_1 = SeparableConvBatchNormBlock(filters=728, kernel_size=3)
        self.sep_conv_batchnorm_mid_2 = SeparableConvBatchNormBlock(filters=728, kernel_size=3)
        self.sep_conv_batchnorm_mid_3 = SeparableConvBatchNormBlock(filters=728, kernel_size=3)

    def call(self, inputs):
        tensor = inputs
        x = self.relu(tensor)
        x = self.sep_conv_batchnorm_mid_1(x)
        x = self.relu(x)
        x = self.sep_conv_batchnorm_mid_2(x)
        x = self.relu(x)
        x = self.sep_conv_batchnorm_mid_3(x)

        tensor = Add()([tensor, x])

        return tensor


class ExitFlow(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(ExitFlow, self).__init__(**kwargs)
        self.num_classes = num_classes

        self.relu = ReLU()
        self.sep_conv_batchnorm_exit_1 = SeparableConvBatchNormBlock(filters=728, kernel_size=3)
        # relu
        self.sep_conv_batchnorm_exit_2 = SeparableConvBatchNormBlock(filters=1024, kernel_size=3)
        # maxpool
        self.maxpool = MaxPool1D(pool_size=3, strides=2, padding='same')

        self.conv_batchnorm_exit_1 = ConvBatchNormBlock(filters=1024, kernel_size=1, strides=2)

        # add
        self.sep_conv_batchnorm_exit_3 = SeparableConvBatchNormBlock(filters=1536, kernel_size=3)
        # relu
        self.sep_conv_batchnorm_exit_4 = SeparableConvBatchNormBlock(filters=2048, kernel_size=3)
        # relu
        # gap
        self.gap = GlobalAveragePooling1D()

        # self.dense = Dense(units=1000, activation='softmax')
        self.dense = Dense(self.num_classes, activation="softmax")

    def call(self, inputs):
        tensor = inputs
        x = self.relu(tensor)
        x = self.sep_conv_batchnorm_exit_1(x)
        x = self.relu(x)
        x = self.sep_conv_batchnorm_exit_2(x)
        x = self.maxpool(x)

        tensor = self.conv_batchnorm_exit_1(tensor)

        x = Add()([tensor, x])

        x = self.sep_conv_batchnorm_exit_3(x)
        x = self.relu(x)
        x = self.sep_conv_batchnorm_exit_4(x)
        x = self.relu(x)
        x = self.gap(x)
        x = self.dense(x)

        return x
