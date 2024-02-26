import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers
from tensorflow import keras
import datetime
import keras_tuner as kt
from ..utils import (
    CustomImageLogging,
    ClassificationMetrics,
    filter_stationary_sequences_dataset,
)

def multi_label_classification(y_true, data_mean, data_std):
    """if for last y in y_true.shape[1]
    y < 80, then [1, 0, 0]
    if 80 <= y < 180, then [0, 1, 0]
    if y >= 180, then [0, 0, 1]


    Args:
        y_true (np.array): array of shape (N, 6)
    """
    y_target = np.zeros((y_true.shape[0], 3))
    # invert the normalization
    y_true = y_true * data_std + data_mean
    # apply the conditions
    y_target = np.where(y_true[:, -1] < 80, 0, 1)
    y_target = np.where(y_true[:, -1] >= 80, 1, y_target)
    y_target = np.where(y_true[:, -1] >= 180, 2, y_target)

    return y_target

def two_label_classification(y_true, data_mean, data_std):
    """if for last y in y_true
    y < 80, then 0
    if y >= 80, then 1

    Args:
        y_true (np.array): array of shape (N, 6)
    """
    y_true = y_true * data_std + data_mean
    # y_target = np.zeros((y_true.shape[0], 1))
    y_target = np.where(y_true[:, -1] < 80, 0, 1)

    return y_target

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model(hp, target="regression"):
    head_size = hp.Int("head_size", min_value=32, max_value=512, step=32)
    num_heads = hp.Int("num_heads", min_value=1, max_value=8, step=1)
    ff_dim = hp.Int("ff_dim", min_value=32, max_value=512, step=32)
    num_transformer_blocks = hp.Int("num_transformer_blocks", min_value=1, max_value=6, step=1)
    mlp_dim = hp.Int("mlp_dim", min_value=32, max_value=512, step=32)
    mlp_layers = hp.Int("mlp_layers", min_value=1, max_value=4, step=1)
    dropout = hp.Float("mlp_dropout", min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])


    inputs = keras.Input(shape=[7, 1])
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for _ in range(mlp_layers):
        x = layers.Dense(mlp_dim, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    if target == "regression":
        outputs = layers.Dense(6)(x)
    elif target == "classification":
        outputs = layers.Dense(1, activation="sigmoid")(x)
    elif target == "multi_classification":
        outputs = layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if target == "regression":
        loss = tf.keras.losses.MeanAbsoluteError()
    elif target == "classification":
        loss = tf.keras.losses.BinaryCrossentropy()
    elif target == "multi_classification":
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        raise ValueError("target must be either regression, classification or multi_classification")

    model.compile(optimizer=optimizer, loss=loss)

    return model

def build_cnn_model(hp, target="regression"):
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
            padding="same",
        )
    )
    for _ in range(n_conv_layers - 1):
        model.add(
            Conv1D(
                filters=filter_units,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )
        )
    # Add flatten layer
    model.add(Flatten())

    # Add dense layers
    for _ in range(n_dense_layers):
        model.add(Dense(dense_units, activation="relu"))

    # Add output layer
    if target == "regression":
        model.add(Dense(6))
    elif target == "classification":
        model.add(Dense(1, activation="sigmoid"))
    elif target == "multi_classification":
        model.add(Dense(3, activation="softmax"))
    # print("CNN model built:", "\n", model.summary())

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if target == "regression":
        loss = tf.keras.losses.MeanAbsoluteError()
    elif target == "classification":
        loss = tf.keras.losses.BinaryCrossentropy()
    elif target == "multi_classification":
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        raise ValueError("target must be either regression, classification or multi_classification")

    model.compile(optimizer=optimizer, loss=loss)

    return model


def build_rnn_model(hp, target="regression"):
    model = Sequential()

    rnn_units = hp.Int("rnn_units", min_value=32, max_value=512, step=32)
    dense_units = hp.Int("dense_units", min_value=32, max_value=512, step=32)
    n_rnn_layers = hp.Int("n_rnn_layers", min_value=1, max_value=6, step=1)
    n_dense_layers = hp.Int("n_dense_layers", min_value=1, max_value=6, step=1)
    learning_rate = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3, 1e-4])

    # Add RNN layers
    for i in range(n_rnn_layers):
        return_sequences = i < (
            n_rnn_layers - 1
        )  # Return sequences for all but the last layer
        model.add(
            tf.keras.layers.LSTM(
                units=rnn_units,
                return_sequences=return_sequences,
            )
        )

    # Add dense layers
    for _ in range(n_dense_layers):
        model.add(Dense(dense_units, activation="relu"))

    # Add output layer
    if target == "regression":
        model.add(Dense(6))
    elif target == "classification":
        model.add(Dense(1, activation="sigmoid"))
    elif target == "multi_classification":
        model.add(Dense(3, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if target == "regression":
        loss = tf.keras.losses.MeanAbsoluteError()
    elif target == "classification":
        print("#########")
        print("classification loss activated")
        loss = tf.keras.losses.BinaryCrossentropy()
    elif target == "multi_classification":
        print("#########")
        print("multi_classification loss activated")
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        raise ValueError("target must be either regression, classification or multi_classification")

    model.compile(optimizer=optimizer, loss=loss)

    return model


def train(
    data_path,
    data_mean,
    data_std,
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
    model_type,
    target,
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
    print("train_y shape:", train_y.shape)
    # if target is classification, transform the target
    if target == "classification":
        train_y = two_label_classification(train_y, data_mean, data_std)
        val_y = two_label_classification(val_y, data_mean, data_std)
        test_y = two_label_classification(test_y, data_mean, data_std)
    elif target == "multi_classification":
        train_y = multi_label_classification(train_y, data_mean, data_std)
        val_y = multi_label_classification(val_y, data_mean, data_std)
        test_y = multi_label_classification(test_y, data_mean, data_std)


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
    EPOCHS = 50

    if model_type == "cnn":
        tuner = kt.BayesianOptimization(
            lambda hp: build_cnn_model(hp, target=target),
            objective="val_loss",
            max_trials=100,
            num_initial_points=None,
            alpha=0.0001,
            beta=2.6,
            directory="logs",
            project_name=f"cnn_optim_{target}",
        )
    elif model_type == "rnn":
        tuner = kt.BayesianOptimization(
            lambda hp: build_rnn_model(hp, target=target),
            objective="val_loss",
            max_trials=100,
            num_initial_points=None,
            alpha=0.0001,
            beta=2.6,
            directory="logs",
            project_name=f"rnn_optim_{target}"
        )
    elif model_type == "transformer":
        tuner = kt.BayesianOptimization(
            lambda hp: build_transformer_model(hp, target=target),
            objective="val_loss",
            max_trials=100,
            num_initial_points=None,
            alpha=0.0001,
            beta=2.6,
            directory="logs",
            project_name=f"transformer_optim_{target}"
        )
    # Using Mean Squared Error for regression tasks

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001
    )
    tuner.search(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[reduce_lr, stop_early],
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hps1 = tuner.results_summary(num_trials=1)

    # save the best hyperparameters to a yaml file for each model_type and target
    with open(f"{model_type}_{target}_best_hps.yaml", "w") as file:
        yaml.dump(best_hps.values, file)

    print(
        f"""
    The hyperparameter search is complete. The best params are:
          {best_hps}
    """
    )


def main():
    # load cnn config from yaml
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # models = ["cnn", "rnn", "transformer"]
    targets = ["regression", "classification", "multi_classification"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="cnn", help="Type of model to train."
    )
    args = parser.parse_args()
    model_type = args.model_type
    print("model_type:", model_type)

    for target_type in targets:
        target = target_type
            
        data_path = config["data_path"]
        data_mean = config["data_mean"]
        data_std = config["data_std"]
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
            data_mean,
            data_std,
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
            model_type,
            target,
        )


if __name__ == "__main__":
    main()
