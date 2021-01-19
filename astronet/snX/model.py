import tensorflow as tf

from astronet.snX.layers import EntryFlow, MiddleFlow, ExitFlow


class SNXModel(tf.keras.Model):
    def __init__(self):
        super(SNXModel, self).__init__()
        '''
        '''
        self.entry_flow = EntryFlow()
        self.middle_flow = [MiddleFlow() for _ in range(8)]
        self.exit_flow = ExitFlow()

    def call(self, inputs, training=None):
        x = self.entry_flow(inputs, training=training)
        for layer in self.middle_flow:
            x = layer(x, training=training)
        output = self.exit_flow(x, training=training)
        return output
