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
from astronet.t2.model import T2Model
from astronet.preprocess import one_hot_encode, tf_one_hot_encode
from astronet.utils import astronet_logger, load_dataset, find_optimal_batch_size

try:
    log = astronet_logger(__file__)
    log.info("=" * shutil.get_terminal_size((80, 20))[0])
    log.info(f"File Path: {Path(__file__).absolute()}")
    log.info(f"Parent of Directory Path: {Path().absolute().parent}")
except:
    print("Seems you are running from a notebook...")
    __file__ = f"{Path().resolve().parent}/astronet/t2/train.py"

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.2f}".format})

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class Training(object):
    # TODO: Update docstrings
    def __init__(self, epochs, dataset, model, redshift, balance):
        self.epochs = EPOCHS
        self.dataset = dataset
        self.model = model
        self.redshift = redshift
        self.balance = balance

    def __call__(self):

        if self.redshift is not None:
            X_train, y_train, X_test, y_test, loss, ZX_train, ZX_test = load_dataset(dataset, redshift=self.redshift)
            hyper_results_file = f"{asnwd}/astronet/t2/opt/runs/{dataset}/results_with_z.json"
        else:
            X_train, y_train, X_test, y_test, loss = load_dataset(dataset, balance=balance)
            hyper_results_file = f"{asnwd}/astronet/t2/opt/runs/{dataset}/results.json"

        num_classes = y_train.shape[1]

        log.info(print(X_train.shape, y_train.shape))

        with open(hyper_results_file) as f:
            events = json.load(f)
            if self.model is not None:
                # Get params for model chosen with cli args
                event = next(item for item in events['optuna_result'] if item["name"] == self.model)
            else:
                event = min(events['optuna_result'], key=lambda ev: ev['objective_score'])

        embed_dim = event['embed_dim']  # --> Embedding size for each token
        num_heads = event['num_heads']  # --> Number of attention heads
        ff_dim = event['ff_dim']  # --> Hidden layer size in feed forward network inside transformer

        # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
        num_filters = embed_dim

        num_samples, timesteps, num_features = X_train.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)
        BATCH_SIZE = find_optimal_batch_size(num_samples)
        print(f"BATCH_SIZE:{BATCH_SIZE}")
        input_shape = (BATCH_SIZE, timesteps, num_features)
        print(f"input_shape:{input_shape}")

        model = T2Model(
            input_dim=input_shape,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_filters=num_filters,
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

        if self.redshift is not None:
            input_shapes = [input_shape, ZX_train.shape]
            model.build_graph(input_shapes)

            train_input = [X_train, ZX_train]
            test_input = [X_test, ZX_test]
        else:
            model.build_graph(input_shape)

            train_input = X_train
            test_input = X_test

        unixtimestamp = int(time.time())
        label = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
        checkpoint_path = f"{asnwd}/astronet/t2/models/{dataset}/model-{unixtimestamp}-{label}"

        history = model.fit(
            train_input,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(test_input, y_test),
            verbose=False,
            callbacks=[
                # DetectOverfittingCallback(threshold=1.5),
                EarlyStopping(
                    min_delta=0.001,
                    mode="min",
                    monitor="val_loss",
                    patience=50,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    mode="min",
                    monitor="val_loss",
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

        print(model.evaluate(test_input, y_test))

        model_params = {}
        model_params['name'] = str(unixtimestamp) + "-" + label
        model_params['hypername'] = event['name']
        model_params['embed_dim'] = event['embed_dim']
        model_params['ff_dim'] = event['ff_dim']
        model_params['num_heads'] = event['num_heads']
        model_params['z-redshift'] = self.redshift
        model_params['balanced_classes'] = self.balance
        # model_params['lr'] = event['lr']
        model_params['model_evaluate_on_test_acc'] = model.evaluate(test_input, y_test)[1]
        model_params['model_evaluate_on_test_loss'] = model.evaluate(test_input, y_test)[0]
        print("  Params: ")
        for key, value in history.history.items():
            print("    {}: {}".format(key, value))
            model_params["{}".format(key)] = value

        del model_params['lr']

        if self.redshift is not None:
            train_results_file = f"{asnwd}/astronet/t2/models/{dataset}/results_with_z.json"
        else:
            train_results_file = f"{asnwd}/astronet/t2/models/{dataset}/results.json"

        with open(train_results_file) as jf:
            data = json.load(jf)
            # print(data)

            previous_results = data['training_result']
            # appending data to optuna_result
            # print(previous_results)
            previous_results.append(model_params)
            # print(previous_results)
            # print(data)

        with open(train_results_file, "w") as rf:
            json.dump(data, rf, sort_keys=True, indent=4)

        model.save(f"{asnwd}/astronet/t2/models/{dataset}/model-{unixtimestamp}-{label}")
        model.save_weights(f"{asnwd}/astronet/t2/models/{dataset}/weights-{unixtimestamp}-{label}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process named model')

    parser.add_argument("-d", "--dataset", default="wisdm_2010",
            help="Choose which dataset to use; options include: 'wisdm_2010', 'wisdm_2019'")

    parser.add_argument("-e", "--epochs", default=20,
            help="How many epochs to run training for")

    parser.add_argument('-m', '--model', default=None,
            help='Name of tensorflow.keras model, i.e. model-<timestamp>-<hash>')

    parser.add_argument('-b', '--balance_classes', default=None,
            help='Use SMOTE or other variant to balance classes')

    parser.add_argument("-z", "--redshift", default=None,
            help="Whether to include redshift features or not")

    try:
        args = parser.parse_args()
        argsdict = vars(args)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    dataset = args.dataset
    EPOCHS = int(args.epochs)
    model = args.model
    balance = args.balance
    redshift = args.redshift
    if redshift is not None:
        redshift = True

    training = Training(
        epochs=EPOCHS, dataset=dataset, model=model, redshift=redshift, balance=balance
    )
    training()
