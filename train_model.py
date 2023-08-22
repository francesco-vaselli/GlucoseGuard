import numpy as np
import yaml
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
import datetime

from utils import CustomImageLogging, ClassificationMetrics


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
):
    # 1. Load the data
    train_x = np.load(data_path)[:, :7]
    train_y = np.load(data_path)[:, 7:]
    print("train_x shape:", train_x.shape)

    # Ensure that train_x has the right shape [samples, timesteps, features]
    # for the moment this stays
    if len(train_x.shape) == 2:
        train_x = np.expand_dims(train_x, axis=-1)

    # 2. Build the dataset
    BATCH_SIZE = batch_size
    BUFFER_SIZE = buffer_size

    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = dataset.take(n_train)
    val_dataset = dataset.skip(n_train).take(n_val)
    test_dataset = dataset.skip(n_train + n_val).take(n_test)

    train_dataset = (
        train_dataset.shuffle(
            dataset.cardinality()
        )  # shuffle all as it is a small dataset
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    # 3. Define the 1D CNN model for regression
    model = build_model(model_config)

    model.compile(
        optimizer="adam", loss="mse"
    )  # Using Mean Squared Error for regression tasks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    # After setting up your train_dataset and val_dataset
    val_samples = next(iter(val_dataset.batch(3)))
    image_logging_callback = CustomImageLogging(log_dir, val_samples)
    # Training the model with reducelronplateau callback and early stopping
    classification_metrics_callback = ClassificationMetrics(
        test_dataset, log_dir, threshold=80
    )
    EPOCHS = epochs
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001
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
