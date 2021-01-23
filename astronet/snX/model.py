import tensorflow as tf

from astronet.snX.layers import EntryFlow, MiddleFlow, ExitFlow


class SNXModel(tf.keras.Model):
    def __init__(self, num_classes, kernel_size):
        super(SNXModel, self).__init__()
        '''
        '''
        self.num_classes = num_classes
        self.kernel_size = kernel_size

        self.entry_flow = EntryFlow(kernel_size=self.kernel_size)
        self.middle_flow = [MiddleFlow(kernel_size=self.kernel_size) for _ in range(8)]
        self.exit_flow = ExitFlow(num_classes=self.num_classes, kernel_size=self.kernel_size)

    def call(self, inputs, training=None):
        x = self.entry_flow(inputs, training=training)
        for layer in self.middle_flow:
            x = layer(x, training=training)
        output = self.exit_flow(x, training=training)
        return output

    def build_graph(self, input_shape):
        # Code lifted from example:
        # https://github.com/tensorflow/tensorflow/issues/29132#issuecomment-504679288
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)
