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

from astronet.t2.model import T2Model
from astronet.t2.preprocess import one_hot_encode
from astronet.t2.utils import t2_logger, load_wisdm_2010, load_wisdm_2019, load_plasticc

try:
    log = t2_logger(__file__)
    log.info("=" * shutil.get_terminal_size((80, 20))[0])
    log.info(f"File Path: {Path(__file__).absolute()}")
    log.info(f"Parent of Directory Path: {Path().absolute().parent}")
except:
    print("Seems you are running from a notebook...")
    __file__ = f"{Path().resolve().parent}/astronet/t2/train.py"

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class Training(object):
    # TODO: Update docstrings
    def __init__(self, epochs, batch_size, dataset):
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        self.dataset = dataset

    def __call__(self):

        if dataset == "wisdm_2010":
            load_dataset = load_wisdm_2010
        elif dataset == "wisdm_2019":
            load_dataset = load_wisdm_2019
        elif dataset == "plasticc":
            load_dataset = load_plasticc

        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
        # One hot encode y
        enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)
        num_classes = y_train.shape[1]

        log.info(print(X_train.shape, y_train.shape))

        with open(f"{Path(__file__).absolute().parent}/opt/runs/{dataset}/results.json") as f:
            events = json.load(f)
            event = max(events['optuna_result'], key=lambda ev: ev['value'])
            print(event)

        embed_dim = event['embed_dim']  # --> Embedding size for each token
        num_heads = event['num_heads']  # --> Number of attention heads
        ff_dim = event['ff_dim']  # --> Hidden layer size in feed forward network inside transformer

        # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
        num_filters = embed_dim

        input_shape = X_train.shape
        # print(input_shape[1:])  # (TIMESTEPS, num_features)

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
            loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=lr), metrics=["acc"]
        )

        model.build_graph(input_shape)

        history = model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            verbose=False,
        )

        model.summary(print_fn=logging.info)

        print(model.evaluate(X_test, y_test))

        unixtimestamp = int(time.time())
        label = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

        model_params = {}
        model_params['name'] = str(unixtimestamp) + "-" + label
        model_params['hypername'] = event['name']
        model_params['embed_dim'] = event['embed_dim']
        model_params['ff_dim'] = event['ff_dim']
        model_params['num_heads'] = event['num_heads']
        model_params['lr'] = event['lr']
        model_params['value'] = model.evaluate(X_test, y_test)[1]
        print("  Params: ")
        for key, value in history.history.items():
            print("    {}: {}".format(key, value))
            model_params["{}".format(key)] = value

        with open(f"{Path(__file__).absolute().parent}/models/{dataset}/results.json") as jf:
            data = json.load(jf)
            print(data)

            previous_results = data['training_result']
            # appending data to optuna_result
            print(previous_results)
            previous_results.append(model_params)
            print(previous_results)
            print(data)

        with open(f"{Path(__file__).absolute().parent}/models/{dataset}/results.json", "w") as rf:
            json.dump(data, rf, sort_keys=True, indent=4)

        model.save(f"{Path(__file__).absolute().parent}/models/{dataset}/model-{unixtimestamp}-{label}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process named model')

    parser.add_argument("-d", "--dataset", default="wisdm_2010",
            help="Choose which dataset to use; options include: 'wisdm_2010', 'wisdm_2019'")

    parser.add_argument("-b", "--batch-size", default=32,
            help="Number of training examples per batch")

    parser.add_argument("-e", "--epochs", default=20,
            help="How many epochs to run training for")

    try:
        args = parser.parse_args()
        argsdict = vars(args)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    dataset = args.dataset
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)

    training = Training(epochs=EPOCHS, batch_size=BATCH_SIZE, dataset=dataset)
    training()
