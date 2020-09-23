import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard.plugins.hparams import api as hp

from astronet.t2.model import T2Model
from astronet.t2.utils import train_val_test_split, create_dataset
from astronet.t2.preprocess import robust_scale, one_hot_encode

from astronet.t2.transformer import TransformerBlock, ConvEmbedding

from pathlib import Path
print("File      Path:", Path(__file__).absolute())
print("Parent of Directory Path:", Path().absolute().parent)


def test_training_pipeline():

    RANDOM_SEED = 42

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Load WISDM-2010 or WISDM-2019 dataset
    column_names = [
        "user_id",
        "activity",
        "timestamp",
        "x_axis",
        "y_axis",
        "z_axis",
    ]

    df = pd.read_csv(str(Path(__file__).absolute().parent.parent.parent.parent.parent) +
        "/data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt",
        header=None,
        names=column_names,
    )
    df.z_axis.replace(regex=True, inplace=True, to_replace=r";", value=r"")
    df["z_axis"] = df.z_axis.astype(np.float64)
    df.dropna(axis=0, how="any", inplace=True)

    print(df.head())

    cols = ["x_axis", "y_axis", "z_axis"]

    print(df[cols].head())

    df_train, df_val, df_test, num_features = train_val_test_split(df, cols)
    print(num_features)  # Should = 3 in this case

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

    print(X_train.shape, y_train.shape)

    # One hot encode y
    enc, y_train, y_val, y_test = one_hot_encode(y_train, y_val, y_test)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    BATCH_SIZE = 32
    EPOCHS = 2

    logdir = "./logs/"

    print(type(X_train))

    embed_dim = 32  # --> Embedding size for each token
    num_heads = 4  # --> Number of attention heads
    ff_dim = 32  # --> Hidden layer size in feed forward network inside transformer

    # --> Number of filters to use in ConvEmbedding block, should be equal to embed_dim
    num_filters = embed_dim

    input_shape = X_train.shape
    print(input_shape[1:])  # (TIMESTEPS, num_features)

    model = T2Model(
        input_dim=input_shape,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_filters=num_filters,
    )

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["acc"]
    )

    _ = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
    )

    model.build_graph(input_shape)

    print(model.summary())

    print(model.evaluate(X_test, y_test))

    result = model.evaluate(X_test, y_test)
    expected_output = [0.44523268938064575, 0.832602322101593]
    assert result[1] == pytest.approx(expected_output[1], 0.1)

# history = model.fit(
#         X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val),
#         callbacks=[
#             tf.keras.callbacks.TensorBoard(logdir),  # log metrics
#             hp.KerasCallback(logdir, hparams),  # log hparams
#             ],)