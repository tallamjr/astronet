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

import argparse
import json
import logging
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import optimizers
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)

from astronet.atx.model import ATXModel
from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.custom_callbacks import DetectOverfittingCallback
from astronet.metrics import WeightedLogLoss
from astronet.utils import (
    astronet_logger,
    find_optimal_batch_size,
    load_dataset,
)

log_filename = f"{asnwd}/astronet/atx/opt/studies.log"

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
except Exception as e:
    print(f"{e}: Seems you are running from a notebook...")
    __file__ = f"{Path().resolve().parent}/astronet/atx/opt/hypertrain.py"

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class Objective(object):
    def __init__(self, epochs, dataset, redshift, augmented, avocado, testset):
        self.epochs = EPOCHS
        self.dataset = dataset
        self.redshift = redshift
        self.augmented = augmented
        self.avocado = avocado
        self.testset = testset

    def __call__(self, trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()

        if self.redshift is not None:
            X_train, y_train, _, _, loss, ZX_train, _ = load_dataset(
                self.dataset,
                redshift=self.redshift,
                augmented=self.augmented,
                avocado=self.avocado,
                testset=self.testset,
            )

            # If redshift true, implies using PLAsTiCC data. PLAsTiCC data is large so we will only
            # work with 10% instead.
            # Generate random boolean mask the length of data
            # use p 0.90 for False and 0.10 for True, i.e down-sample by 90%
            mask = np.random.choice([False, True], len(X_train), p=[0.90, 0.10])
            X_train = X_train[mask]
            y_train = y_train[mask]
            ZX_train = ZX_train[mask]
            log.info("Dataset downsampled by 90% for cross-validation steps..")

        elif self.dataset == "plasticc":
            X_train, y_train, _, _, loss = load_dataset(
                self.dataset,
                redshift=self.redshift,
                augmented=self.augmented,
                avocado=self.avocado,
                testset=self.testset,
            )
            # Again, if using PLAsTiCC data, the PLAsTiCC data is large so we will only
            # work with 10% instead.
            # Generate random boolean mask the length of data
            # use p 0.90 for False and 0.10 for True, i.e down-sample by 90%
            mask = np.random.choice([False, True], len(X_train), p=[0.90, 0.10])
            X_train = X_train[mask]
            y_train = y_train[mask]
            log.info("Dataset downsampled by 90% for cross-validation steps..")

        else:
            X_train, y_train, _, _, loss = load_dataset(
                self.dataset,
                redshift=self.redshift,
                augmented=self.augmented,
                avocado=self.avocado,
                testset=self.testset,
            )

        num_classes = y_train.shape[1]

        kernel_size = trial.suggest_categorical(
            "kernel_size", [3, 16, 32, 48]
        )  # --> Filter length
        pool_size = trial.suggest_categorical(
            "pool_size", [3, 16, 32, 48]
        )  # --> Pooling width
        scaledown_factor = trial.suggest_categorical(
            "scaledown_factor", [4]
        )  # --> Reduce number of filters down by given factor

        (
            num_samples,
            timesteps,
            num_features,
        ) = X_train.shape  # X_train.shape[1:] == (TIMESTEPS, num_features)
        BATCH_SIZE = find_optimal_batch_size(num_samples)
        print(f"BATCH_SIZE:{BATCH_SIZE}")
        input_shape = (BATCH_SIZE, timesteps, num_features)
        print(f"input_shape:{input_shape}")

        model = ATXModel(
            num_classes=num_classes,
            kernel_size=kernel_size,
            pool_size=pool_size,
            scaledown_factor=scaledown_factor,
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
        # 'random_state' has no effect since shuffle is False. You should leave random_state to its default
        # (None), or set shuffle=True.'
        k_folds = 5
        skf = StratifiedKFold(n_splits=k_folds, shuffle=False, random_state=None)

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

        kth_fold = 1
        for train_index, val_index in skf.split(X_train, y_train_split):
            X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
            y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

            inputs_train_cv = X_train_cv
            inputs_val_cv = X_val_cv

            if self.redshift is not None:
                Z_train_cv, Z_val_cv = ZX_train[train_index], ZX_train[val_index]
                inputs_train_cv = [X_train_cv, Z_train_cv]
                inputs_val_cv = [X_val_cv, Z_val_cv]

            VALIDATION_BATCH_SIZE = find_optimal_batch_size(X_val_cv.shape[0])
            print(f"VALIDATION_BATCH_SIZE:{VALIDATION_BATCH_SIZE}")

            _ = model.fit(
                inputs_train_cv,
                y_train_cv,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(inputs_val_cv, y_val_cv),
                validation_batch_size=VALIDATION_BATCH_SIZE,
                verbose=False,
                callbacks=[
                    # DetectOverfittingCallback(
                    #     threshold=2
                    # ),
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
            log.info(f"{kth_fold} of {k_folds} k-folds complete")
            kth_fold += 1

            # Evaluate the model accuracy on the validation set.
            # loss, _ = model.evaluate(inputs_val_cv, y_val_cv, verbose=0, batch_size=VALIDATION_BATCH_SIZE)
            wloss = WeightedLogLoss()
            y_preds = model.predict(inputs_val_cv, batch_size=VALIDATION_BATCH_SIZE)
            loss = wloss(y_val_cv, y_preds).numpy()
            scores.append(loss)

            log.info(f"Intermediate loss score: {loss}")

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
    try:
        label = (
            subprocess.check_output(["git", "describe", "--always"]).strip().decode()
        )
    except Exception:
        from astronet import __version__ as current_version

        label = current_version

    parser = argparse.ArgumentParser(description="Optimising hyperparameters")

    parser.add_argument(
        "-d",
        "--dataset",
        default="wisdm_2010",
        help="Choose which dataset to use; options include: 'wisdm_2010', 'wisdm_2019'",
    )

    parser.add_argument(
        "-e", "--epochs", default=10, help="How many epochs to run training for"
    )

    parser.add_argument(
        "-n",
        "--num-trials",
        default=15,
        help="Number of trials to run optimisation. Each trial will have N-epochs, where N equals args.epochs",
    )

    parser.add_argument(
        "-z",
        "--redshift",
        default=None,
        help="Whether to include redshift features or not",
    )

    parser.add_argument(
        "-a", "--augment", default=None, help="Train using augmented plasticc data"
    )

    parser.add_argument(
        "-A",
        "--avocado",
        default=None,
        help="Train using avocado augmented plasticc data",
    )

    parser.add_argument(
        "-t",
        "--testset",
        default=None,
        help="Train using PLAsTiCC test data for representative test",
    )

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

    augmented = args.augment
    if augmented is not None:
        augmented = True

    avocado = args.avocado
    if avocado is not None:
        avocado = True

    testset = args.testset
    if testset is not None:
        testset = True

    N_TRIALS = int(args.num_trials)

    study = optuna.create_study(study_name=f"{unixtimestamp}", direction="minimize")

    study.optimize(
        Objective(
            epochs=EPOCHS,
            dataset=dataset,
            redshift=redshift,
            augmented=augmented,
            avocado=avocado,
            testset=testset,
        ),
        n_trials=N_TRIALS,
        timeout=86400,  # Break out of optimisation after ~ 24 hrs
        n_jobs=-1,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    log.warn(
        """show_progress_bar: Flag to show progress bars \n
        "or not. To disable progress bar, set this ``False``.  Currently, \n
        progress bar is experimental feature and disabled when \n
        ``n_jobs`` != 1`."""
    )

    best_result = {}
    best_result["name"] = str(unixtimestamp) + "-" + label

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    df_study = study.trials_dataframe()
    print(df_study.head())

    print("  Value: {}".format(trial.value))
    best_result["objective_score"] = trial.value

    best_result["z-redshift"] = redshift
    best_result["augmented"] = augmented
    best_result["avocado"] = avocado
    best_result["testset"] = testset

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        # best_result["{}".format(key)] = value

    best_result.update(study.best_params)
    print(best_result)

    if redshift:
        hyper_results_file = (
            f"{asnwd}/astronet/atx/opt/runs/{dataset}/results_with_z.json"
        )
    else:
        hyper_results_file = f"{asnwd}/astronet/atx/opt/runs/{dataset}/results.json"

    with open(hyper_results_file) as jf:
        data = json.load(jf)
        print(data)

        previous_results = data["optuna_result"]
        # Appending data to optuna_result
        print(previous_results)
        previous_results.append(best_result)
        print(previous_results)
        print(data)

    with open(hyper_results_file, "w") as rf:
        json.dump(data, rf, sort_keys=True, indent=4)

    with open(
        f"{asnwd}/astronet/atx/opt/runs/{dataset}/study-{unixtimestamp}-{label}.pkl",
        "wb",
    ) as sf:
        joblib.dump(study, sf)
