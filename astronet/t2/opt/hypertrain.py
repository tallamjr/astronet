import argparse
import joblib
import json
import logging
import numpy as np
import optuna
import os
import shutil
import subprocess
import sys
import tensorflow as tf
import warnings

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import optimizers
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)

from astronet.constants import astronet_working_directory as asnwd
from astronet.custom_callbacks import DetectOverfittingCallback
from astronet.metrics import WeightedLogLoss
from astronet.t2.model import T2Model
from astronet.preprocess import one_hot_encode, tf_one_hot_encode
from astronet.utils import astronet_logger, load_dataset, find_optimal_batch_size

try:
    print(os.environ['ASNWD'])
    log_filename = str(os.environ['ASNWD']) + "/astronet/t2/opt/studies.log"
except KeyError:
    print("Please set the environment ASNWD in 'conf/astronet.conf'")
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(filename=log_filename, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

try:
    log = astronet_logger(__file__)
    log.info("=" * shutil.get_terminal_size((80, 20))[0])
    log.info(f"File Path: {Path(__file__).absolute()}")
    log.info(f"Parent of Directory Path: {Path().absolute().parent}")
except:
    print("Seems you are running from a notebook...")
    __file__ = f"{Path().resolve().parent}/astronet/t2/opt/hypertrain.py"

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class Objective(object):
    def __init__(self, epochs, dataset, redshift, balance):
        self.epochs = EPOCHS
        self.dataset = dataset
        self.redshift = redshift
        self.balance = balance

    def __call__(self, trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()

        if self.redshift is not None:
            X_train, y_train, _, _, loss, ZX_train, _ = load_dataset(dataset, redshift=self.redshift)
        else:
            X_train, y_train, _, _, loss = load_dataset(dataset, balance=self.balance)

        num_classes = y_train.shape[1]

        embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128, 512])  # --> Embedding size for each token
        num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])  # --> Number of attention heads
        ff_dim = trial.suggest_categorical("ff_dim", [32, 64, 128, 512])  # --> Hidden layer size in feed forward network inside transformer

        num_filters = embed_dim  # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
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
        lr = trial.suggest_float("lr", 1e-2, 1e-1, log=True)
        model.compile(
            loss=loss,
            optimizer=optimizers.Adam(lr=lr, clipnorm=1),
            metrics=["acc"],
            # Allows for values to be show when debugging
            # Also required for use with custom_log_loss
            run_eagerly=True,
        )

        scores = []
        skf = StratifiedKFold(n_splits=5, random_state=RANDOM_SEED)

        if self.redshift is not None:
            num_z_samples, num_z_features = ZX_train.shape
            z_input_shape = (BATCH_SIZE, num_z_features)
            input_shapes = [input_shape, z_input_shape]
            model.build_graph(input_shapes)
        else:
            model.build_graph(input_shape)

        print(type(y_train))
        if tf.is_tensor(y_train):
            y_train = np.array(y_train)
            print(type(y_train))
            y_train_split = y_train.argmax(1)
            print(y_train_split)
        else:
            y_train_split = y_train.argmax(1)
            print(y_train_split)

        for train_index, val_index in skf.split(X_train, y_train_split):
            X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
            y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

            inputs_train_cv = X_train_cv
            inputs_val_cv = X_val_cv

            if self.redshift is not None:
                Z_train_cv, Z_val_cv = ZX_train[train_index], ZX_train[val_index]
                inputs_train_cv = [X_train_cv, Z_train_cv]
                inputs_val_cv = [X_val_cv, Z_val_cv]

            _ = model.fit(
                inputs_train_cv,
                y_train_cv,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(inputs_val_cv, y_val_cv),
                verbose=False,
                callbacks=[
                    # DetectOverfittingCallback(threshold=1.5),
                    EarlyStopping(
                        min_delta=0.001,
                        mode="min",
                        monitor="val_loss",
                        patience=10,
                        restore_best_weights=True,
                        verbose=1,
                    ),
                    ReduceLROnPlateau(
                        cooldown=5,
                        factor=0.1,
                        mode="min",
                        monitor="val_loss",
                        patience=5,
                        verbose=1,
                    ),
                ],
            )

            # Evaluate the model accuracy on the validation set.
            loss, _ = model.evaluate(inputs_val_cv, y_val_cv, verbose=0)
            scores.append(loss)

        model.summary(print_fn=logging.info)
        return np.mean(scores)


if __name__ == "__main__":
    warnings.warn(
        "Recent Keras release (2.4.0) simply redirects all APIs "
        "in the standalone keras package to point to tf.keras. "
        "There is now only one Keras: tf.keras. "
        "There may be some breaking changes for some workflows by upgrading to keras 2.4.0. "
        "Test before upgrading. "
        "REF:https://github.com/keras-team/keras/releases/tag/2.4.0"
    )

    import time
    unixtimestamp = int(time.time())
    label = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

    parser = argparse.ArgumentParser(description='Optimising hyperparameters')

    parser.add_argument("-d", "--dataset", default="wisdm_2010",
            help="Choose which dataset to use; options include: 'wisdm_2010', 'wisdm_2019'")

    parser.add_argument("-e", "--epochs", default=10,
            help="How many epochs to run training for")

    parser.add_argument("-n", "--num-trials", default=15,
            help="Number of trials to run optimisation. Each trial will have N-epochs, where N equals args.epochs")

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

    redshift = args.redshift
    if redshift is not None:
        redshift = True

    balance = args.balance
    if balance is not None:
        balance = True

    N_TRIALS = int(args.num_trials)

    study = optuna.create_study(study_name=f"{unixtimestamp}", direction="minimize")

    study.optimize(
        Objective(epochs=EPOCHS, dataset=dataset, redshift=redshift, balance=balance),
        n_trials=N_TRIALS,
        timeout=86400,
        n_jobs=-1,
        show_progress_bar=False,
    )

    log.warn("""show_progress_bar: Flag to show progress bars \n
        "or not. To disable progress bar, set this ``False``.  Currently, \n
        progress bar is experimental feature and disabled when \n
        ``n_jobs`` != 1`.""")

    best_result = {}
    best_result['name'] = str(unixtimestamp) + "-" + label

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    df_study = study.trials_dataframe()
    print(df_study.head())

    print("  Value: {}".format(trial.value))
    best_result['objective_score'] = trial.value

    best_result['z-redshift'] = redshift
    best_result['balanced_classes'] = balance

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        # best_result["{}".format(key)] = value

    best_result.update(study.best_params)
    print(best_result)

    if redshift:
        hyper_results_file = f"{asnwd}/astronet/t2/opt/runs/{dataset}/results_with_z.json"
    else:
        hyper_results_file = f"{asnwd}/astronet/t2/opt/runs/{dataset}/results.json"

    with open(hyper_results_file) as jf:
        data = json.load(jf)
        print(data)

        previous_results = data['optuna_result']
        # Appending data to optuna_result
        print(previous_results)
        previous_results.append(best_result)
        print(previous_results)
        print(data)

    with open(hyper_results_file, "w") as rf:
        json.dump(data, rf, sort_keys=True, indent=4)

    with open(f"{asnwd}/astronet/t2/opt/runs/{dataset}/study-{unixtimestamp}-{label}.pkl", "wb") as sf:
        joblib.dump(study, sf)
