from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import numpy as np
import tensorflow as tf


class DenseLayerTest(tf.test.TestCase):
    def setUp(self):
        super(DenseLayerTest, self).setUp()
        self.my_dense = tf.keras.layers.Dense(2)
        self.my_dense.build((2, 2))

    def testDenseLayerOutput(self):
        self.my_dense.set_weights([np.array([[1, 0], [2, 3]]), np.array([0.5, 0])])
        input_x = np.array([[1, 2], [2, 3]])
        output = self.my_dense(input_x)
        with self.subTest():
            expected_output = np.array([[5.5, 6.0], [8.5, 9]])
            self.assertAllEqual(expected_output, output)


# tf.test.main()
