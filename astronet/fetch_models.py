import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Union

import numpy as np
import psutil
import tensorflow as tf

from astronet.atx.model import ATXModel
from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.constants import SYSTEM
from astronet.t2.model import T2Model
from astronet.tinho.funcmodel import build_model


def fetch_model(
    model: str,
    hyper_results_file: str,
    input_shapes: Union[list, tuple],
    architecture: str = "t2",
    num_classes: int = 14,
):
    """
    # A list for input_shapes would imply there is multiple inputs
    """
    with open(hyper_results_file) as f:
        events = json.load(f)
        if model is not None:
            # Get params for model chosen with cli args
            event = next(
                item for item in events["optuna_result"] if item["name"] == model
            )
        else:
            event = min(events["optuna_result"], key=lambda ev: ev["objective_score"])

    # A list would imply there is multiple inputs, therefore dealing with additional features,
    # i.e. redshift etc
    if isinstance(input_shapes, list):
        input_shape = input_shapes[0]
        num_aux_feats = input_shapes[1][1]  # Take Z_train.shape[1]
    else:
        input_shape = input_shapes
        num_aux_feats = 0

    if architecture == "tinho":
        embed_dim = event["embed_dim"]  # --> Embedding size for each token
        num_heads = event["num_heads"]  # --> Number of attention heads
        ff_dim = event[
            "ff_dim"
        ]  # --> Hidden layer size in feed forward network inside transformer

        # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
        num_filters = embed_dim

        num_layers = event["num_layers"]  # --> N x repeated transformer blocks
        droprate = event["droprate"]  # --> Rate of neurons to drop
        # fc_neurons = event['fc_neurons']    # --> N neurons in final Feed forward network.

        model = build_model(
            input_shapes,
            input_dim=input_shape,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_filters=num_filters,
            num_classes=num_classes,
            num_layers=num_layers,
            droprate=droprate,
            num_aux_feats=num_aux_feats,
            add_aux_feats_to="L",
            # Either add features to M dimension or L dimension. Adding to L allows for
            # visualisation of CAMs relating to redshift since we would have a CAM of (L + Z) x c
            # fc_neurons=fc_neurons,
        )

    elif architecture == "t2":

        embed_dim = event["embed_dim"]  # --> Embedding size for each token
        num_heads = event["num_heads"]  # --> Number of attention heads
        ff_dim = event[
            "ff_dim"
        ]  # --> Hidden layer size in feed forward network inside transformer

        # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
        num_filters = embed_dim

        num_layers = event["num_layers"]  # --> N x repeated transformer blocks
        droprate = event["droprate"]  # --> Rate of neurons to drop
        # fc_neurons = event['fc_neurons']    # --> N neurons in final Feed forward network.

        model = T2Model(
            input_dim=input_shape,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_filters=num_filters,
            num_classes=num_classes,
            num_layers=num_layers,
            droprate=droprate,
            num_aux_feats=num_aux_feats,
            add_aux_feats_to="L",
            # Either add features to M dimension or L dimension. Adding to L allows for
            # visualisation of CAMs relating to redshift since we would have a CAM of (L + Z) x c
            # fc_neurons=fc_neurons,
        )

        model.build_graph(input_shapes)

    elif architecture == "atx":

        kernel_size = event["kernel_size"]  # --> Filter length
        pool_size = event["pool_size"]  # --> Pooling width
        scaledown_factor = event[
            "scaledown_factor"
        ]  # --> Reduce number of filters down by given factor

        model = ATXModel(
            num_classes=num_classes,
            kernel_size=kernel_size,
            pool_size=pool_size,
            scaledown_factor=scaledown_factor,
        )

        model.build_graph(input_shapes)

    return model, event
