import numpy as np
import argparse
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, LSTM
from tensorflow.keras.callbacks import TensorBoard
import datetime

from utils import (
    CustomImageLogging,
    ClassificationMetrics,
    filter_stationary_sequences_dataset,
)


def build_model(model_config):
    if model_config["model_type"] == "cnn":
        cnn_config = model_config["cnn_config"]
        model = Sequential()

        # Add convolutional layers
        for _ in range(cnn_config["n_conv_layers"]):
            model.add(
                Conv1D(
                    filters=cnn_config["filters"],
                    kernel_size=cnn_config["kernel_size"],
                    activation=cnn_config["activation"],
                    input_shape=cnn_config["input_shape"],
                )
            )

        # Add flatten layer
        model.add(Flatten())

        # Add dense layers
        for _ in range(cnn_config["n_dense_layers"]):
            model.add(
                Dense(cnn_config["dense_size"], activation=cnn_config["activation"])
            )

        # Add output layer
        model.add(Dense(cnn_config["output_shape"]))
        print("CNN model built:", "\n", model.summary())

    elif model_config["model_type"] == "rnn":
        rnn_config = model_config["rnn_config"]
        model = Sequential()
        # Add RNN layers
        for i in range(rnn_config["n_rnn_layers"]):
            return_sequences = i < (rnn_config["n_rnn_layers"] - 1)
            model.add(
                LSTM(
                    rnn_config["rnn_units"],
                    return_sequences=return_sequences,
                    # input_shape=rnn_config["input_shape"],
                )
                # Or you can use GRU: GRU(rnn_config["units"], return_sequences=return_sequences)
            )

        # Add dense layers
        for _ in range(rnn_config["n_dense_layers"]):
            model.add(
                Dense(rnn_config["dense_size"], activation="relu")
            )

        # Add output layer
        model.add(Dense(rnn_config["output_shape"]))
        print("RNN model done")
    else:
        raise NotImplementedError(f"{model_config['model_type']} not implemented")

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
    log_name,
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

    # 3. Define the 1D CNN model for regression
    model = build_model(model_config)
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise NotImplementedError(f"{optimizer} not implemented")

    if loss == "mse":
        loss = tf.keras.losses.MeanSquaredError()
    elif loss == "mae":
        loss = tf.keras.losses.MeanAbsoluteError()
    else:
        raise NotImplementedError(f"{loss} not implemented")

    model.compile(
        optimizer=optimizer, loss=loss
    )  # Using Mean Squared Error for regression tasks
    log_dir = "logs/fit/" + log_name
    # datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    # After setting up your train_dataset and val_dataset
    image_logging_callback = CustomImageLogging(log_dir, val_dataset)
    # Training the model with reducelronplateau callback and early stopping
    classification_metrics_callback = ClassificationMetrics(
        test_dataset, log_dir, test_y=test_y, threshold=80
    )
    EPOCHS = epochs
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
            tensorboard_callback,
            image_logging_callback,
            classification_metrics_callback,
        ],
    )
    # plot the loss

    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.legend()
    plt.savefig("loss.png")


def main():
    # parse command line arguments to get log name

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_name", type=str, required=True)
    args = parser.parse_args()
    log_name = args.log_name
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
        log_name,
    )


if __name__ == "__main__":
    main()
