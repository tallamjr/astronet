import joblib
import json
import logging
import numpy as np
import optuna
import subprocess
import sys
import tensorflow as tf
import warnings

from pathlib import Path
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.backend import clear_session

from astronet.t2.model import T2Model
from astronet.t2.preprocess import one_hot_encode
from astronet.t2.transformer import TransformerBlock, ConvEmbedding
from astronet.t2.utils import t2_logger, load_WISDM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename='studies.log', mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
)

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

try:
    log = t2_logger(__file__)
    log.info("_________________________________")
    log.info("File Path:" + str(Path(__file__).absolute()))
    log.info("Parent of Directory Path:" + str(Path().absolute().parent))
except:
    print("Seems you are running from a notebook...")
    __file__ = str(Path().resolve().parent) + "/astronet/t2/opt/hypertrain.py"

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class Objective(object):
    def __init__(self, epochs, batch_size):
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE

    def __call__(self, trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()

        # Load WISDM-2010
        X_train, y_train, X_val, y_val, X_test, y_test = load_WISDM()
        # One hot encode y
        enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)

        embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128, 512])  # --> Embedding size for each token
        num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])  # --> Number of attention heads
        ff_dim = trial.suggest_categorical("ff_dim", [32, 64, 128, 512])  # --> Hidden layer size in feed forward network inside transformer

        num_filters = embed_dim  # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim

        input_shape = X_train.shape
        # print(input_shape[1:])  # --> (TIMESTEPS, num_features)

        model = T2Model(
            input_dim=input_shape,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_filters=num_filters,
        )

        # We compile our model with a sampled learning rate.
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        model.compile(
            loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=lr), metrics=["acc"]
        )

        model.build_graph(input_shape)

        _ = model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            verbose=False,
        )

        model.summary(print_fn=logging.info)

        # Evaluate the model accuracy on the validation set.
        # score = model.evaluate(X_val, y_val, verbose=0)
        score = model.evaluate(X_test, y_test, verbose=0)
        return score[1]


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

    BATCH_SIZE = 32
    EPOCHS = 20
    N_TRIALS = 20

    study = optuna.create_study(study_name=f"{unixtimestamp}", direction="maximize")
    study.optimize(Objective(epochs=EPOCHS, batch_size=BATCH_SIZE), n_trials=N_TRIALS, timeout=1000)

    best_result = {}
    best_result['name'] = str(unixtimestamp) + "-" + label

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    df_study = study.trials_dataframe()
    print(df_study.head())

    print("  Value: {}".format(trial.value))
    best_result['value'] = trial.value

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        # best_result["{}".format(key)] = value

    best_result.update(study.best_params)
    print(best_result)

    with open(f"{Path(__file__).absolute().parent}/runs/results.json") as jf:
        data = json.load(jf)
        print(data)

        previous_results = data['optuna_result']
        # Appending data to optuna_result
        print(previous_results)
        previous_results.append(best_result)
        print(previous_results)
        print(data)

    with open(f"{Path(__file__).absolute().parent}/runs/results.json", "w") as rf:
        json.dump(data, rf, sort_keys=True, indent=4)

    with open(f"{Path(__file__).absolute().parent}/runs/study-{unixtimestamp}-{label}.pkl", "wb") as sf:
        joblib.dump(study, sf)
