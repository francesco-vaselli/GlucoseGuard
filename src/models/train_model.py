import numpy as np
import argparse
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, LSTM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers
from tensorflow import keras
import datetime

from param_scan import multi_label_classification, two_label_classification
from src.utils import (
    CustomImageLogging,
    ClassificationMetrics,
    filter_stationary_sequences_dataset,
    train_transfer,
)


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


def build_attn_model(
    input_shape,
    output_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(output_shape)(x)
    return keras.Model(inputs, outputs)


class FeedBack(tf.keras.Model):
    def __init__(
        self, units, unit_size, out_steps, num_features, dense_units, dense_size
    ):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cells = tf.keras.layers.LSTMCell(unit_size)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cells, return_state=True)
        self.dense_layers = [
            tf.keras.layers.Dense(dense_size, activation="relu")
            for _ in range(dense_units)
        ]
        self.output_layer = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        prediction = self.output_layer(x)
        return prediction, state

    def call(self, inputs, training=None):
        predictions = []
        prediction, state = self.warmup(inputs)

        predictions.append(prediction)

        for n in range(1, self.out_steps):
            # x = prediction
            x = tf.expand_dims(prediction, 1)
            x, *state = self.lstm_rnn(
                x, initial_state=state, training=training
            )  # Note: using lstm_rnn here

            for dense_layer in self.dense_layers:
                x = dense_layer(x)

            prediction = self.output_layer(x)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        predictions = tf.reshape(predictions, [-1, self.out_steps])

        return predictions


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
        print("CNN model built:", "\n")
        model.summary()

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
                    input_shape=rnn_config["input_shape"],
                )
                # Or you can use GRU: GRU(rnn_config["units"], return_sequences=return_sequences)
            )

        # Add dense layers
        for _ in range(rnn_config["n_dense_layers"]):
            model.add(Dense(rnn_config["dense_size"], activation="relu"))

        # Add output layer
        model.add(Dense(rnn_config["output_shape"]))
        print("RNN model done")
        model.summary()
    elif model_config["model_type"] == "ar_rnn":
        ar_rnn_config = model_config["ar_rnn_config"]
        model = FeedBack(
            units=ar_rnn_config["units"],
            unit_size=ar_rnn_config["unit_size"],
            out_steps=ar_rnn_config["out_steps"],
            num_features=ar_rnn_config["num_features"],
            dense_units=ar_rnn_config["dense_units"],
            dense_size=ar_rnn_config["dense_size"],
        )
        print("AR-RNN model built:", "\n")
        # feedback_model.summary()
    elif model_config["model_type"] == "attn":
        attn_config = model_config["attn_config"]
        model = build_attn_model(
            input_shape=attn_config["input_shape"],
            output_shape=attn_config["output_shape"],
            head_size=attn_config["head_size"],
            num_heads=attn_config["num_heads"],
            ff_dim=attn_config["ff_dim"],
            num_transformer_blocks=attn_config["num_transformer_blocks"],
            mlp_units=attn_config["mlp_units"],
            dropout=attn_config["dropout"],
            mlp_dropout=attn_config["mlp_dropout"],
        )
        print("Transformer model built:", "\n")
        model.summary()

    else:
        raise NotImplementedError(f"{model_config['model_type']} not implemented")

    return model


def train(
    data_path,
    data_mean,
    data_std,
    transfer_data_path,
    transfer_data_mean,
    transfer_data_std,
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
    target="regression",
    save_model=True,
    transfer_learning=False,
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

    # 3. Define the 1D CNN model for regression
    model = build_model(model_config)
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise NotImplementedError(f"{optimizer} not implemented")

    if target == "regression":
        loss = tf.keras.losses.MeanAbsoluteError()
    elif target == "multi_classification":
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    elif target == "classification":
        loss = tf.keras.losses.BinaryCrossentropy()
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
        test_dataset, log_dir, test_y=test_y, threshold=80, std=data_std, mean=data_mean
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

    if save_model:
        model.save(f"models/{log_name}.h5")

    if transfer_learning:
        model, history = train_transfer(
            transfer_data_path,
            transfer_data_mean,
            transfer_data_std,
            log_name,
            batch_size=32,
            epochs=epochs,
            model=model,
        )


def main():
    # parse command line arguments to get log name

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_name", type=str, required=True)
    args = parser.parse_args()
    log_name = args.log_name
    # load cnn config from yaml
    with open("configs/train_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = config["data_path"]
    data_mean = config["data_mean"]
    data_std = config["data_std"]
    transfer_data_path = config["transfer_data_path"]
    transfer_data_mean = data_mean
    transfer_data_std = data_std
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
    save_model = config["save_model"]
    transfer_learning = config["transfer_learning"]

    train(
        data_path,
        data_mean,
        data_std,
        transfer_data_path,
        transfer_data_mean,
        transfer_data_std,
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
        save_model,
        transfer_learning,
    )


if __name__ == "__main__":
    main()
