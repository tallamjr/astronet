import argparse
import json
import logging
import numpy as np
import shutil
import subprocess
import sys
import tensorflow as tf
import time

from pathlib import Path
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from astronet.constants import astronet_working_directory as asnwd
from astronet.custom_callbacks import DetectOverfittingCallback
from astronet.metrics import WeightedLogLoss
from astronet.snX.model import SNXModel
from astronet.preprocess import one_hot_encode, tf_one_hot_encode
from astronet.utils import astronet_logger, load_dataset, find_optimal_batch_size

try:
    log = astronet_logger(__file__)
    log.info("=" * shutil.get_terminal_size((80, 20))[0])
    log.info(f"File Path: {Path(__file__).absolute()}")
    log.info(f"Parent of Directory Path: {Path().absolute().parent}")
except:
    print("Seems you are running from a notebook...")
    __file__ = f"{Path().resolve().parent}/astronet/snX/train.py"

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class Training(object):
    # TODO: Update docstrings
    def __init__(self, epochs, dataset):
        self.epochs = EPOCHS
        self.dataset = dataset

    def __call__(self):

        X_train, y_train, X_test, y_test, loss = load_dataset(dataset)

        num_classes = y_train.shape[1]

        log.info(print(X_train.shape, y_train.shape))

        with open(f"{asnwd}/astronet/snX/opt/runs/{dataset}/results.json") as f:
            events = json.load(f)
            event = min(events['optuna_result'], key=lambda ev: ev['objective_score'])

        num_samples, timesteps, num_features = X_train.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)
        BATCH_SIZE = find_optimal_batch_size(num_samples)
        print(f"BATCH_SIZE:{BATCH_SIZE}")
        input_shape = (BATCH_SIZE, timesteps, num_features)
        print(f"input_shape:{input_shape}")

        model = SNXModel(
            num_classes=num_classes,
        )

        # We compile our model with a sampled learning rate.
        lr = event['lr']
        model.compile(
            loss=loss,
            optimizer=optimizers.Adam(lr=lr, clipnorm=1),
            metrics=["acc"],
            run_eagerly=True,  # Show values when debugging. Also required for use with custom_log_loss
        )

        model.build_graph(input_shape)

        unixtimestamp = int(time.time())
        label = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
        checkpoint_path = f"{asnwd}/astronet/snX/models/{dataset}/model-{unixtimestamp}-{label}"

        history = model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            verbose=False,
            callbacks=[
                # DetectOverfittingCallback(threshold=1.5),
                EarlyStopping(
                    min_delta=0.001,
                    mode="min",
                    monitor="loss",
                    patience=50,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    mode="min",
                    monitor="loss",
                    save_best_only=True,
                ),
                ReduceLROnPlateau(
                    cooldown=5,
                    factor=0.1,
                    mode="min",
                    monitor="loss",
                    patience=5,
                    verbose=1,
                ),
            ],
        )

        model.summary(print_fn=logging.info)

        print(model.evaluate(X_test, y_test))

        model_params = {}
        model_params['name'] = str(unixtimestamp) + "-" + label
        model_params['hypername'] = event['name']
        model_params['model_evaluate_on_test_acc'] = model.evaluate(X_test, y_test)[1]
        model_params['model_evaluate_on_test_loss'] = model.evaluate(X_test, y_test)[0]
        print("  Params: ")
        for key, value in history.history.items():
            print("    {}: {}".format(key, value))
            model_params["{}".format(key)] = value

        del model_params['lr']

        with open(f"{asnwd}/astronet/snX/models/{dataset}/results.json") as jf:
            data = json.load(jf)
            # print(data)

            previous_results = data['training_result']
            # appending data to optuna_result
            # print(previous_results)
            previous_results.append(model_params)
            # print(previous_results)
            # print(data)

        with open(f"{asnwd}/astronet/snX/models/{dataset}/results.json", "w") as rf:
            json.dump(data, rf, sort_keys=True, indent=4)

        model.save(f"{asnwd}/astronet/snX/models/{dataset}/model-{unixtimestamp}-{label}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process named model')

    parser.add_argument("-d", "--dataset", default="wisdm_2010",
            help="Choose which dataset to use; options include: 'wisdm_2010', 'wisdm_2019'")

    parser.add_argument("-e", "--epochs", default=20,
            help="How many epochs to run training for")

    try:
        args = parser.parse_args()
        argsdict = vars(args)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    dataset = args.dataset
    EPOCHS = int(args.epochs)

    training = Training(epochs=EPOCHS, dataset=dataset)
    training()
