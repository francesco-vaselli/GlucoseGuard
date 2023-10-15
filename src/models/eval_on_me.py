from tensorflow.keras.models import load_model
import numpy as np
import argparse
import yaml
import os
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
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, auc
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Your existing evaluation functions like plot_confusion_matrix, plot_roc_curve, check_classification...
def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    # plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def check_classification(
    true, pred, threshold=80, standard=True, ind=5, std=57.94, mean=144.98
):
    # Assuming true and pred have shape [batch_size, seq_length, feature_dim]
    # and that the value of interest is the last in the sequence

    if standard:
        pred = pred * std + mean
        true = true * std + mean
    else:
        pred *= 100
        true *= 100

    pred_label = (pred[:, ind] < threshold).astype(int)  # Assuming feature_dim = 1
    true_label = (true[:, ind] < threshold).astype(int)  # Adjust index if different

    fpr, tpr, _ = roc_curve(true_label, pred_label)
    roc_auc = auc(fpr, tpr)

    tn, fp, fn, tp = confusion_matrix(true_label, pred_label).ravel()
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    f1 = tp / (tp + 1 / 2 * (fp + fn))

    return (
        true_label,
        pred_label,
        fpr,
        tpr,
        roc_auc,
        accuracy,
        sensitivity,
        specificity,
        precision,
        npv,
        f1,
    )

def plot_beautiful_fig(x, y_true, y_pred, title, save_path, mean, std):
    # multiply by std and add mean
    x = x * std + mean
    y_true = y_true * std + mean
    y_pred = y_pred * std + mean

    time_intervals_x = np.arange(0, 5 * x[0].shape[0], 5)
    time_intervals_y = np.arange(
        5 * x[0].shape[0], 5 * x[0].shape[0] + 5 * y_true[0].shape[0], 5
    )
    # cat x and y_true
    x = np.concatenate((x, y_true), axis=1)
    time_intervals_full = np.concatenate((time_intervals_x, time_intervals_y))

    # Create figures and log them
    for i in range(x.shape[0]):
        # specify the figure size
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the actual data points with circle markers

        ax.scatter(
            time_intervals_full,
            x[i, :],
            c="blue",
            label="Actual Measurements",
            marker="o",
        )

        # Plot the predicted data points with circle markers

        ax.scatter(
            time_intervals_y,
            y_pred[i, :],
            c="red",
            label="Predicted Measurements",
            marker="x",
        )

        # add vertical line at 31 minutes and write the text "Sampling Time" on left, "Prediction Time" on right
        # also add arrows pointing left and right
        ax.axvline(x=31, color="black", linestyle="--")
        ax.text(0.35, 0.85, "Sampling \n Horizon", transform=ax.transAxes, fontsize=15)
        ax.text(
            0.55, 0.85, "Prediction \n Horizon", transform=ax.transAxes, fontsize=15
        )

        # Add labels and title

        ax.set_xlabel("Time (minutes)", fontsize=15)

        ax.set_ylabel("Blood Glucose Level (mg/dL)", fontsize=15)

        ax.set_title("Blood Glucose Level Over Time", fontsize=15)

        ax.legend(loc="upper left", fontsize=15)

        # save
        fig.savefig(f"{save_path}{title}_{i}.png")

def eval():

    log_name = "attn_transf_good"  # Replace this with the name you used while saving the model
    model_path = f"models/{log_name}.h5"
    loaded_model = load_model(model_path)  # If you have custom layers
    print(loaded_model.input_shape)

    mean = 144.98
    std = 57.94
    new_test_x = [95, 105, 113, 121, 122, 122, 124]
    new_test_y = [133, 140, 151, 160, 164, 163]
    # standardize
    new_test_x = (new_test_x - mean) / std
    new_test_y = (new_test_y - mean) / std

    new_test_x = np.array(new_test_x).reshape(None, 1, 7)
    new_test_y = np.array(new_test_y)
    print("train_x shape:", new_test_x.shape)

    new_test_dataset = tf.data.Dataset.from_tensor_slices((new_test_x, new_test_y))


    log_dir = "logs/on_me/" + log_name
    os.makedirs(log_dir, exist_ok=True)

    pred_y = loaded_model.predict(new_test_x)

    # # compute MAE and RMSE
    # mae = mean_absolute_error(new_test_y, pred_y)
    # rmse = mean_squared_error(new_test_y, pred_y, squared=False)
    # print(f"for {log_name} MAE: {mae}, RMSE: {rmse}")

    # # get back to original scale
    # mae = mae * 57.941
    # rmse = rmse * 57.941
    # print(f"for {log_name} scaled MAE: {mae}, scaled RMSE: {rmse}")

    # # get values for comparison with other paper
    # scale_paper = 60.565/57.941
    # mae = mae * scale_paper
    # rmse = rmse * scale_paper
    # print(f"for {log_name} paper scaled MAE: {mae}, paper scaled RMSE: {rmse}")

    # # now we do the same MAE and RMSE but only for the last point
    # mae = mean_absolute_error(new_test_y[:, -1], pred_y[:, -1])
    # rmse = mean_squared_error(new_test_y[:, -1], pred_y[:, -1], squared=False)
    # print(f"for {log_name} last point MAE: {mae}, last point RMSE: {rmse}")

    # # get back to original scale
    # mae = mae * 57.941
    # rmse = rmse * 57.941
    # print(f"for {log_name} last point scaled MAE: {mae}, last point scaled RMSE: {rmse}")
    
    # # get values for comparison with other paper
    # mae = mae * scale_paper
    # rmse = rmse * scale_paper
    # print(f"for {log_name} paper scaled last point MAE: {mae}, paper scaled last point RMSE: {rmse}")

    new_test_x = new_test_x.reshape(-1, 7)

    plot_beautiful_fig(new_test_x[:3], new_test_y[:3], pred_y[:3], "new_test", log_dir, 144.98, 57.94)
    (
        new_test_y,
        pred_y,
        fpr,
        tpr,
        roc_auc,
        accuracy,
        sensitivity,
        specificity,
        precision,
        npv,
        f1,
    ) = check_classification(
        new_test_y,
        pred_y,
        threshold=80,
        std=57.94,
        mean=144.98,
    )
    cm = confusion_matrix(new_test_y, pred_y)
    cm_fig = plot_confusion_matrix(cm, class_names=["Hyper", "Hypo"])
    # save cm_fig
    plt.savefig(log_dir+"/confusion_matrix_svm.png")
    with open(log_dir+"/metrics.txt", "w") as f:
        f.write(
            f"Accuracy: {accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\nPrecision: {precision}\nNPV: {npv}\nF1: {f1}"
        )

if __name__ == "__main__":
    eval()