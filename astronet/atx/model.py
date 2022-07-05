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

from astronet.atx.layers import EntryFlow, ExitFlow, MiddleFlow


class ATXModel(tf.keras.Model):
    def __init__(self, num_classes, kernel_size, pool_size, scaledown_factor):
        super(ATXModel, self).__init__()
        """
        """
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.scaledown_factor = scaledown_factor

        self.entry_flow = EntryFlow(
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
            scaledown_factor=self.scaledown_factor,
        )

        self.middle_flow = [
            MiddleFlow(
                kernel_size=self.kernel_size, scaledown_factor=self.scaledown_factor
            )
            for _ in range(1)
        ]

        self.exit_flow = ExitFlow(
            num_classes=self.num_classes,
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
            scaledown_factor=self.scaledown_factor,
        )

    def call(self, inputs, training=None):
        # If not a list then inputs are of type tensor: tf.is_tensor(inputs) == True
        if tf.is_tensor(inputs):
            x = self.entry_flow(inputs, training=training)
            for layer in self.middle_flow:
                x = layer(x, training=training)
            output = self.exit_flow(x, training=training)
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
            x = self.entry_flow(x, training=training)
            for layer in self.middle_flow:
                x = layer(x, training=training)
            output = self.exit_flow([x, z], training=training)

        return output

    def model(self):
        x = tf.keras.Input(shape=(100, 6))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    def build_graph(self, input_shapes):
        if isinstance(
            input_shapes, tuple
        ):  # A list would imply there is multiple inputs
            # Code lifted from example:
            # https://github.com/tensorflow/tensorflow/issues/29132#issuecomment-504679288
            input_shape_nobatch = input_shapes[1:]
            self.build(input_shapes)
            inputs = tf.keras.Input(shape=input_shape_nobatch)
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
