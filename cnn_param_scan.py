import numpy as np
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
import datetime
import keras_tuner as kt

from utils import (
    CustomImageLogging,
    ClassificationMetrics,
    filter_stationary_sequences_dataset,
)


def build_model(hp):
    model = Sequential()

    filter_units = hp.Int("filter_units", min_value=32, max_value=512, step=32)
    kernel_size = hp.Choice("kernel_size", values=[1, 2, 3])
    dense_units = hp.Int("dense_units", min_value=32, max_value=512, step=32)
    n_conv_layers = hp.Int("n_conv_layers", min_value=1, max_value=6, step=1)
    n_dense_layers = hp.Int("n_dense_layers", min_value=1, max_value=6, step=1)
    learning_rate = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])

    # Add convolutional layers

    model.add(
        Conv1D(
            filters=filter_units,
            kernel_size=kernel_size,
            activation="relu",
            input_shape=[7, 1],
        )
    )
    for _ in range(n_conv_layers-1):
        model.add(
            Conv1D(
                filters=filter_units,
                kernel_size=kernel_size,
                activation="relu",
            )
        )
    # Add flatten layer
    model.add(Flatten())

    # Add dense layers
    for _ in range(n_dense_layers):
        model.add(Dense(dense_units, activation="relu"))

    # Add output layer
    model.add(Dense(6))
    # print("CNN model built:", "\n", model.summary())

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss = tf.keras.losses.MeanAbsoluteError()

    model.compile(optimizer=optimizer, loss=loss)

    return model


def train(
    data_path,
    n_train,
    n_val,
    n_test,
    batch_size,
    buffer_size,
    epochs,
    optimizer,
    loss,
    learning_rate,
    model_config,
):
    # 1. Load the data
    ds = np.load(data_path)
    ds = filter_stationary_sequences_dataset(ds)
    train_x = ds[:n_train, :7]
    train_y = ds[:n_train, 7:]
    val_x = ds[n_train : n_train + n_val, :7]
    val_y = ds[n_train : n_train + n_val, 7:]
    test_x = ds[n_train + n_val : n_train + n_val + n_test, :7]
    test_y = ds[n_train + n_val : n_train + n_val + n_test, 7:]
    print("train_x shape:", train_x.shape)

    # Ensure that train_x has the right shape [samples, timesteps, features]
    # for the moment this stays
    if len(train_x.shape) == 2:
        train_x = np.expand_dims(train_x, axis=-1)
        val_x = np.expand_dims(val_x, axis=-1)
        test_x = np.expand_dims(test_x, axis=-1)

    # 2. Build the dataset
    BATCH_SIZE = batch_size
    BUFFER_SIZE = buffer_size

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    train_dataset = (
        train_dataset.shuffle(
            train_dataset.cardinality()
        )  # shuffle all as it is a small dataset
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    val_dataset = val_dataset.batch(BATCH_SIZE)
    # get one element from the val dataset and print its shape
    print("val_dataset shape:", next(iter(val_dataset.batch(1)))[0].shape)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    EPOCHS = epochs

    # Using Mean Squared Error for regression tasks
    tuner = kt.Hyperband(
        build_model,
        objective="val_loss",
        max_epochs=EPOCHS,
        factor=3,
        directory="logs",
        project_name="cnn_optim",
    )
    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001
    )
    tuner.search(train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[stop_early, reduce_lr])

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The best params are:
          {best_hps}
    """)


def main():
    # load cnn config from yaml
    with open("train_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = config["data_path"]
    n_train = config["n_train"]
    n_val = config["n_val"]
    n_test = config["n_test"]
    batch_size = config["batch_size"]
    buffer_size = config["buffer_size"]
    epochs = config["epochs"]
    optimizer = config["optimizer"]
    loss = config["loss"]
    learning_rate = config["learning_rate"]
    model_config = config["model_config"]

    train(
        data_path,
        n_train,
        n_val,
        n_test,
        batch_size,
        buffer_size,
        epochs,
        optimizer,
        loss,
        learning_rate,
        model_config,
    )


if __name__ == "__main__":
    main()
