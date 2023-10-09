from tensorflow.keras.models import load_model
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

from src.utils import (
    CustomImageLogging,
    ClassificationMetrics,
    filter_stationary_sequences_dataset,
    train_transfer,
)

def eval():

    log_name = "attn_transf_good"  # Replace this with the name you used while saving the model
    model_path = f"models/{log_name}.h5"
    loaded_model = load_model(model_path)  # If you have custom layers

    data_path = "data/dataset_ohio_smooth_stdbyref.npy"
    # 1. Load the data
    ds = np.load(data_path)
    ds = filter_stationary_sequences_dataset(ds)

    new_test_x = ds[:, :7].reshape(-1, 7, 1)
    new_test_y = ds[:, 7:].reshape(-1, 1)
    print("train_x shape:", new_test_x.shape)

    new_test_dataset = tf.data.Dataset.from_tensor_slices((new_test_x, new_test_y))


    log_dir = "logs/new_test/" + log_name
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    image_logging_callback = CustomImageLogging(log_dir, new_test_dataset)
    classification_metrics_callback = ClassificationMetrics(
        new_test_dataset, log_dir, test_y=new_test_y, threshold=80, std=144.98, mean=57.94
    )

    # model.evaluate() or model.predict()
    loaded_model.evaluate(
        new_test_dataset,
        callbacks=[
            tensorboard_callback,
            image_logging_callback,
            classification_metrics_callback
        ]
    )

if __name__ == "__main__":
    eval()