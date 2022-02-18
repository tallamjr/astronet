import os

import numpy as np
import tensorflow as tf

from astronet.t2.multihead_attention import (
    MultiHeadAttention,
    scaled_dot_product_attention,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class MultiheadAttentionTest(tf.test.TestCase):
    def testScaledDotProductAttentionOutputCorrectness(self):
        batch_size = 1
        seq_len = 2
        depth = 1
        q = tf.ones([batch_size, seq_len, depth])
        k = tf.ones([batch_size, seq_len, depth]) + 1
        v = tf.ones([batch_size, seq_len, depth]) + 1
        output, attention_weights, _ = scaled_dot_product_attention(
            q, k, v, mask=None, debug_mode=True
        )

        with self.subTest():
            expected_output = np.array([[[2.0], [2.0]]])  # shape: (1, 2, 1)
            expected_attention_weight = np.array([[[0.5, 0.5], [0.5, 0.5]]])

            self.assertAllEqual(expected_output, output)
            self.assertAllEqual(expected_attention_weight, attention_weights)

    def testScaledDotProductAttentionTensorShapes(self):
        batch_size = 1
        seq_len = 2
        depth = 1
        q = tf.ones([batch_size, seq_len, depth])
        k = tf.ones([batch_size, seq_len, depth]) + 1
        v = tf.ones([batch_size, seq_len, depth]) + 1
        output, attention_weights, debug_dict = scaled_dot_product_attention(
            q, k, v, mask=None, debug_mode=True
        )

        with self.subTest():
            # Create dummy zeros arrays with expected dimensions
            self.assertShapeEqual(
                np.zeros((batch_size, seq_len, seq_len)), debug_dict["matmul_qk"]
            )  #
            self.assertShapeEqual(
                np.zeros(()), debug_dict["dk"]
            )  # 0 is a dummy value with shape ()
            self.assertShapeEqual(
                np.zeros((batch_size, seq_len, seq_len)),
                debug_dict["scaled_attention_logits"],
            )
            self.assertShapeEqual(
                np.zeros((batch_size, seq_len, seq_len)),
                debug_dict["attention_weights"],
            )
            self.assertShapeEqual(np.zeros((batch_size, seq_len, depth)), output)
            self.assertShapeEqual(
                np.zeros((batch_size, seq_len, seq_len)), attention_weights
            )

    def testScaledDotProductAttentionMask(self):
        batch_size = 2
        seq_len = 3
        depth = 1
        np.random.seed(10)
        q = np.random.rand(batch_size, seq_len, depth).astype(
            "float32"
        )  # (batch_size, seq_len, depth)
        k = np.random.rand(batch_size, seq_len, depth).astype("float32")
        v = np.random.rand(batch_size, seq_len, depth).astype("float32")

        mask = create_look_ahead_mask(3)

        flip_zero_one_func = lambda x: 0 if x == 1 else 1
        flip_zero_one_func = np.vectorize(flip_zero_one_func)
        flipped_mask = flip_zero_one_func(mask)

        output, attention_weights, _ = scaled_dot_product_attention(
            q, k, v, mask=mask, debug_mode=True
        )

        with self.subTest():
            # compare batch by batch. We have batch size 2 in this test method.
            for attention_weights_one_batch in attention_weights:
                # flatten two tensors, so that we can compare them in one loop.
                expected = tf.reshape(flipped_mask, [-1])
                output = tf.reshape(attention_weights_one_batch, [-1])
                # Use a zip to pair expected value and output value.
                for expected_val, output_val in zip(expected, output):
                    self.assertAllGreaterEqual(expected_val, output_val)

    def testMultiHeadAttentionOutputShapes(self):
        d_model = 4
        num_heads = 2
        batch_size = 2
        seq_len = 4
        depth = d_model // num_heads

        # def call(self, v, k, q, mask):
        multihead_attention = MultiHeadAttention(d_model, num_heads, debug_mode=True)

        q = tf.ones([batch_size, seq_len, depth])
        k = tf.ones([batch_size, seq_len, depth])
        v = tf.ones([batch_size, seq_len, depth])

        output, _, debug_dict = multihead_attention(v, k, q, mask=None)

        with self.subTest():

            self.assertShapeEqual(
                np.zeros((batch_size, num_heads, seq_len, depth)),
                debug_dict["split_q"],
            )
            self.assertShapeEqual(
                np.zeros((batch_size, num_heads, seq_len, depth)),
                debug_dict["split_k"],
            )
            self.assertShapeEqual(
                np.zeros((batch_size, num_heads, seq_len, depth)),
                debug_dict["split_v"],
            )
            self.assertShapeEqual(
                np.zeros((batch_size, seq_len, num_heads, depth)),
                debug_dict["scaled_attention"],
            )
            self.assertShapeEqual(
                np.zeros((batch_size, seq_len, d_model)),
                debug_dict["concat_attention"],
            )
            self.assertShapeEqual(np.zeros((batch_size, seq_len, d_model)), output)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
