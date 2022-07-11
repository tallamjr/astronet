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
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers


class ClusterableWeightsCA(tfmot.clustering.keras.ClusteringAlgorithm):
    """This class provides a special lookup function for the the weights 'w'.
    It reshapes and tile centroids the same way as the weights. This allows us
    to find pulling indices efficiently.
    """

    def get_pulling_indices(self, weight):
        clst_num = self.cluster_centroids.shape[0]
        tiled_weights = tf.tile(tf.expand_dims(weight, axis=2), [1, 1, clst_num])
        tiled_cluster_centroids = tf.tile(
            tf.reshape(self.cluster_centroids, [1, 1, clst_num]),
            [weight.shape[0], weight.shape[1], 1],
        )

        # We find the nearest cluster centroids and store them so that ops can build
        # their kernels upon it
        pulling_indices = tf.argmin(
            tf.abs(tiled_weights - tiled_cluster_centroids), axis=2
        )

        return pulling_indices


class PrunableClusterableLayer(
    tf.keras.layers.Layer,
    tfmot.sparsity.keras.PrunableLayer,
    tfmot.clustering.keras.ClusterableLayer,
):
    def get_prunable_weights(self):
        # Prune bias also, though that usually harms model accuracy too much.
        return [("kernel", self.kernel)]

    def get_clusterable_weights(self):
        # Cluster kernel and bias. This is just an example, clustering
        # bias usually hurts model accuracy.
        return [("kernel", self.kernel), ("bias", self.bias)]

    def get_clusterable_algorithm(self, weight_name):
        """Returns clustering algorithm for the custom weights 'w'."""
        if weight_name == "kernel":
            return ClusterableWeightsCA
        else:
            # We don't cluster other weights.
            return None


class ConvEmbedding(PrunableClusterableLayer):
    def __init__(self, num_filters, **kwargs):
        super(ConvEmbedding, self).__init__(**kwargs)
        self.num_filters = num_filters

        self.conv1d = layers.Conv1D(
            filters=num_filters, kernel_size=1, activation="relu"
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_filters": self.num_filters,
            }
        )
        return config

    def call(self, inputs):
        embedding = self.conv1d(inputs)

        return embedding


class ClusterWeights(PrunableClusterableLayer):
    def __init__(self, num_classes=14, number_of_clusters=16, **kwargs):
        super(ClusterWeights, self).__init__()
        self.num_classes = num_classes

        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

        self.number_of_clusters = number_of_clusters
        self.cluster_centroids_init = CentroidInitialization.LINEAR

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
            }
        )
        return config

    def call(self, inputs):

        cluster_weights = tfmot.clustering.keras.cluster_weights

        clustered_dense = cluster_weights(
            tf.keras.layers.Dense(self.num_classes, activation="softmax"),
            self.number_of_clusters,
            self.cluster_centroids_init,
        )(inputs)

        return clustered_dense
