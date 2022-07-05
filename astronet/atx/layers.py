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
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    GlobalAveragePooling1D,
    MaxPool1D,
    ReLU,
    SeparableConv1D,
)


class ConvBatchNormBlock(tf.keras.Model):
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


class SeparableConvBatchNormBlock(tf.keras.Model):
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


class EntryFlow(tf.keras.Model):
    def __init__(self, kernel_size, pool_size, scaledown_factor, **kwargs):
        super(EntryFlow, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.scaledown_factor = scaledown_factor

        self.conv_batchnorm_1 = ConvBatchNormBlock(
            filters=32 / self.scaledown_factor, kernel_size=self.kernel_size, strides=2
        )
        self.relu = ReLU()
        self.conv_batchnorm_2 = ConvBatchNormBlock(
            filters=64 / self.scaledown_factor, kernel_size=self.kernel_size
        )
        # relu

        self.sep_conv_batchnorm_1 = SeparableConvBatchNormBlock(
            filters=128 / self.scaledown_factor, kernel_size=self.kernel_size
        )
        # relu
        self.sep_conv_batchnorm_2 = SeparableConvBatchNormBlock(
            filters=128 / self.scaledown_factor, kernel_size=self.kernel_size
        )
        # relu
        # maxpool

        self.conv_batchnorm_3 = ConvBatchNormBlock(
            filters=128 / self.scaledown_factor, kernel_size=1, strides=2
        )

        # add
        # relu

        self.sep_conv_batchnorm_4 = SeparableConvBatchNormBlock(
            filters=256 / self.scaledown_factor, kernel_size=self.kernel_size
        )
        # relu
        self.sep_conv_batchnorm_5 = SeparableConvBatchNormBlock(
            filters=256 / self.scaledown_factor, kernel_size=self.kernel_size
        )
        # maxpool

        self.conv_batchnorm_4 = ConvBatchNormBlock(
            filters=256 / self.scaledown_factor, kernel_size=1, strides=2
        )

        # add
        # relu

        self.sep_conv_batchnorm_6 = SeparableConvBatchNormBlock(
            filters=728 / self.scaledown_factor, kernel_size=self.kernel_size
        )
        # relu
        self.sep_conv_batchnorm_7 = SeparableConvBatchNormBlock(
            filters=728 / self.scaledown_factor, kernel_size=self.kernel_size
        )
        # maxpool
        self.maxpool = MaxPool1D(pool_size=self.pool_size, strides=2, padding="same")

        self.conv_batchnorm_5 = ConvBatchNormBlock(
            filters=728 / self.scaledown_factor, kernel_size=1, strides=2
        )
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


class MiddleFlow(tf.keras.Model):
    def __init__(self, kernel_size, scaledown_factor, **kwargs):
        super(MiddleFlow, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.scaledown_factor = scaledown_factor

        self.relu = ReLU()
        self.sep_conv_batchnorm_mid_1 = SeparableConvBatchNormBlock(
            filters=728 / self.scaledown_factor, kernel_size=self.kernel_size
        )
        self.sep_conv_batchnorm_mid_2 = SeparableConvBatchNormBlock(
            filters=728 / self.scaledown_factor, kernel_size=self.kernel_size
        )
        self.sep_conv_batchnorm_mid_3 = SeparableConvBatchNormBlock(
            filters=728 / self.scaledown_factor, kernel_size=self.kernel_size
        )

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


class ExitFlow(tf.keras.Model):
    def __init__(self, num_classes, kernel_size, pool_size, scaledown_factor, **kwargs):
        super(ExitFlow, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.scaledown_factor = scaledown_factor

        self.relu = ReLU()
        self.sep_conv_batchnorm_exit_1 = SeparableConvBatchNormBlock(
            filters=728 / self.scaledown_factor, kernel_size=self.kernel_size
        )
        # relu
        self.sep_conv_batchnorm_exit_2 = SeparableConvBatchNormBlock(
            filters=1024 / self.scaledown_factor, kernel_size=self.kernel_size
        )
        # maxpool
        self.maxpool = MaxPool1D(pool_size=self.pool_size, strides=2, padding="same")

        self.conv_batchnorm_exit_1 = ConvBatchNormBlock(
            filters=1024 / self.scaledown_factor, kernel_size=1, strides=2
        )

        # add
        self.sep_conv_batchnorm_exit_3 = SeparableConvBatchNormBlock(
            filters=1536 / self.scaledown_factor, kernel_size=self.kernel_size
        )
        # relu
        self.sep_conv_batchnorm_exit_4 = SeparableConvBatchNormBlock(
            filters=2048 / self.scaledown_factor, kernel_size=self.kernel_size
        )
        # relu
        # gap
        self.gap = GlobalAveragePooling1D()

        # self.dense = Dense(units=1000, activation='softmax')
        self.dense = Dense(self.num_classes, activation="softmax")

    def call(self, inputs):
        # If not a list then inputs are of type tensor: tf.is_tensor(inputs) == True
        if tf.is_tensor(inputs):
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
        # Else this implies input is a list; a list of tensors, i.e. multiple inputs
        else:
            # X in L x M
            x = inputs[0]
            # Additional Z features
            z = inputs[1]
            # >>> z.shape
            # TensorShape([None, 2])

            tensor = x
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

            # Concatenate redshift information to 2048 vector --> 2050
            x = tf.keras.layers.Concatenate(axis=1)([x, z])
            # >>> x.shape
            # TensorShape([None, 2050])

            x = self.dense(x)

        return x
