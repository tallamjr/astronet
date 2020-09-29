import joblib
import json
import logging
import optuna
import subprocess
import sys
import warnings

from pathlib import Path

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

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras.backend import clear_session
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorboard.plugins.hparams import api as hp

from astronet.t2.model import T2Model
from astronet.t2.utils import train_val_test_split, create_dataset
from astronet.t2.preprocess import robust_scale, one_hot_encode

from astronet.t2.transformer import TransformerBlock, ConvEmbedding

from pathlib import Path
print("File      Path:", Path(__file__).absolute())
print("Parent of Directory Path:", Path().absolute().parent)


RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    # Load WISDM-2010 or WISDM-2019 dataset
    column_names = [
        "user_id",
        "activity",
        "timestamp",
        "x_axis",
        "y_axis",
        "z_axis",
    ]

    df = pd.read_csv(str(Path(__file__).absolute().parent.parent.parent.parent) +
        "/data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",
        header=None,
        names=column_names,
    )
    df.z_axis.replace(regex=True, inplace=True, to_replace=r";", value=r"")
    df["z_axis"] = df.z_axis.astype(np.float64)
    df.dropna(axis=0, how="any", inplace=True)

    # print(df.head())

    cols = ["x_axis", "y_axis", "z_axis"]

    # print(df[cols].head())

    df_train, df_val, df_test, num_features = train_val_test_split(df, cols)
    # print(num_features)  # Should = 3 in this case

    # Perfrom robust scaling
    robust_scale(df_train, df_val, df_test, cols)

    TIME_STEPS = 200
    STEP = 40

    X_train, y_train = create_dataset(
        df_train[cols],
        df_train.activity,
        TIME_STEPS,
        STEP
    )

    X_val, y_val = create_dataset(
        df_val[cols],
        df_val.activity,
        TIME_STEPS,
        STEP
    )

    X_test, y_test = create_dataset(
        df_test[cols],
        df_test.activity,
        TIME_STEPS,
        STEP
    )

    # print(X_train.shape, y_train.shape)

    # One hot encode y
    enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)

    # print(X_train.shape, y_train.shape)
    # print(X_val.shape, y_val.shape)
    # print(X_test.shape, y_test.shape)

    BATCH_SIZE = 32
    EPOCHS = 2

    # logdir = "./logs/"

    # print(type(X_train))

    # embed_dim = 32  # --> Embedding size for each token
    # num_heads = 4  # --> Number of attention heads
    # ff_dim = 32  # --> Hidden layer size in feed forward network inside transformer

    embed_dim = trial.suggest_categorical("embed_dim", [32, 64])  # --> Embedding size for each token
    num_heads = trial.suggest_categorical("num_heads", [4, 8])  # --> Number of attention heads
    ff_dim = trial.suggest_categorical("ff_dim", [32, 64])  # --> Hidden layer size in feed forward network inside transformer

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

    # model.build(input_shape)

    model.summary(print_fn=logging.info)
    # print(model.evaluate(X_test, y_test))

    # Evaluate the model accuracy on the validation set.
    # score = model.evaluate(X_val, y_val, verbose=0)
    score = model.evaluate(X_test, y_test, verbose=0)
    return score[1]
    # history = model.fit(
    #         X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val),
    #         callbacks=[
    #             tf.keras.callbacks.TensorBoard(logdir),  # log metrics
    #             hp.KerasCallback(logdir, hparams),  # log hparams
    #             ],)


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

    study = optuna.create_study(study_name="{unixtimestamp}", direction="maximize")
    study.optimize(objective, n_trials=3, timeout=600)

    best_result = {}
    best_result['name'] = str(unixtimestamp) + "-" + label

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    best_result['best_trial'] = trial

    print("  Value: {}".format(trial.value))
    best_result['value'] = trial.value

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        best_result["{}".format(key)] = value

    best_result.update(study.best_params)

    with open(f"{Path().absolute()}/runs/results.json") as jf:
        data = json.load(jf)

        previous_results = data['optuna_result']
        # appending data to optuna_result
        previous_results.append(best_result)

    with open(f"{Path().absolute()}/runs/result.json", "w") as rf:
        json.dump(data, rf, sort_keys=True, indent=4)

    with open(f"{Path().absolute()}/runs/study-{unixtimestamp}-{label}.pkl", "wb") as sf:
        joblib.dump(study, sf)
