import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard.plugins.hparams import api as hp

from astronet.t2.model import T2Model
from astronet.t2.preprocess import train_val_test_split, create_dataset

# Load WISDM-2010 or WISDM-2019 dataset

BATCH_SIZE=32
EPOCHS=20

logdir = "./logs/"

TIME_STEPS = 200
STEP = 40

X_train, y_train = create_dataset(
    df_train[['x_axis', 'y_axis', 'z_axis']],
    df_train.activity,
    TIME_STEPS,
    STEP
)

model = T2Model(X_train)

history = model.fit(
        X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.TensorBoard(logdir),  # log metrics
            hp.KerasCallback(logdir, hparams),  # log hparams
            ],)

