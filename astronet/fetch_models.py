import argparse
import copy
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
from astronet.utils import astronet_logger

log = astronet_logger(__file__)


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

    model_params = copy.deepcopy(event)

    popkeys = [
        "augmented",
        "avocado",
        "lr",
        "name",
        "objective_score",
        "testset",
        "z-redshift",
    ]
    for key in event:
        if key in popkeys:
            model_params.pop(key)

    log.info(model_params)

    if architecture == "tinho":

        num_filters = event["embed_dim"]  # --> Embedding size for each token

        model = build_model(
            input_shapes,
            num_filters=num_filters,
            input_dim=input_shape,
            num_aux_feats=num_aux_feats,
            add_aux_feats_to="L",
            num_classes=num_classes,
            **model_params,
        )

    elif architecture == "t2":

        num_filters = event["embed_dim"]  # --> Embedding size for each token

        model = T2Model(
            input_dim=input_shape,
            num_filters=num_filters,
            num_aux_feats=num_aux_feats,
            add_aux_feats_to="L",
            num_classes=num_classes,
            **model_params,
        )
        model.build_graph(input_shapes)

    elif architecture == "atx":

        model = ATXModel(num_classes=num_classes, **model_params)
        model.build_graph(input_shapes)

    return model, event
