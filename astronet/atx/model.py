import tensorflow as tf

from astronet.atx.layers import EntryFlow, MiddleFlow, ExitFlow


class ATXModel(tf.keras.Model):
    def __init__(self, num_classes, kernel_size, pool_size):
        super(ATXModel, self).__init__()
        '''
        '''
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        self.entry_flow = EntryFlow(kernel_size=self.kernel_size, pool_size=self.pool_size)
        self.middle_flow = [MiddleFlow(kernel_size=self.kernel_size) for _ in range(8)]
        self.exit_flow = ExitFlow(
            num_classes=self.num_classes, kernel_size=self.kernel_size, pool_size=self.pool_size
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

    def build_graph(self, input_shape):
        if isinstance(input_shape, tuple):  # A list would imply there is multiple inputs
            # Code lifted from example:
            # https://github.com/tensorflow/tensorflow/issues/29132#issuecomment-504679288
            input_shape_nobatch = input_shape[1:]
            self.build(input_shape)
            inputs = tf.keras.Input(shape=input_shape_nobatch)
        else:
            input_shape_nobatch = input_shapes[0][1:]
            Z_input_shape_nobatch = input_shapes[1][1:]
            inputs = [
                tf.keras.Input(shape=input_shape_nobatch),
                tf.keras.Input(shape=Z_input_shape_nobatch),
            ]

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)
