import io
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools


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
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def plot_roc_curve(fpr, tpr, roc_auc):
    figure, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    return figure


def check_classification(true, pred, threshold=80, standard=True, ind=5):
    # Assuming true and pred have shape [batch_size, seq_length, feature_dim]
    # and that the value of interest is the last in the sequence

    std, mean = 58.119, 144.982
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

    return true_label, pred_label, fpr, tpr, roc_auc


class ClassificationMetrics(tf.keras.callbacks.Callback):
    def __init__(self, val_data, log_dir, threshold=80):
        super().__init__()
        self.val_data = val_data
        self.threshold = threshold
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if (
            epoch % 5 != 4
        ):  # Here we check if the epoch is a multiple of 5 (adjust if 0-indexing is confusing)
            return
        x, y_true = self.val_data
        y_pred = self.model.predict(x)

        true_label, pred_label, fpr, tpr, roc_auc = check_classification(
            y_true, y_pred, self.threshold
        )

        with self.writer.as_default():
            tf.summary.scalar("ROC AUC", roc_auc, step=epoch)

            # You can add more metrics if desired, e.g.
            tn, fp, fn, tp = confusion_matrix(true_label, pred_label).ravel()
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            sensitivity = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = tp / (tp + 1 / 2 * (fp + fn))
            tf.summary.scalar("Accuracy", accuracy, step=epoch)
            tf.summary.scalar("Sensitivity", sensitivity, step=epoch)
            tf.summary.scalar("Precision", precision, step=epoch)
            tf.summary.scalar("F1", f1, step=epoch)

            # Log ROC curve
            figure = plot_roc_curve(fpr, tpr, roc_auc)
            tf.summary.image("ROC Curve", self.plot_to_image(figure), step=epoch)

            cm = confusion_matrix(true_label, pred_label)
            figure = plot_confusion_matrix(
                cm, class_names=["Above Threshold", "Below Threshold"]
            )
            tf.summary.image("Confusion Matrix", self.plot_to_image(figure), step=epoch)

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        buf = io.BytesIO()

        # Use plt.savefig to save the plot to a PNG in memory.
        plt.savefig(buf, format="png")
        plt.close(figure)
        buf.seek(0)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Expand the dimensions to [1, *, *, 4]
        image = tf.expand_dims(image, 0)

        return image


class CustomImageLogging(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, val_dataset, num_samples=3):
        super().__init__()
        self.log_dir = log_dir
        self.val_data = val_dataset
        self.num_samples = num_samples
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        # if (
        #     epoch % 5 != 4
        # ):  # Here we check if the epoch is a multiple of 5 (adjust if 0-indexing is confusing)
        #    return
        # Get predictions
        x, y_true = self.val_data
        y_pred = self.model.predict(x[: self.num_samples])
        time_intervals = np.arange(0, 5 * x[0].shape[0], 5)

        # Create figures and log them
        for i in range(self.num_samples):
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot input time series
            ax.plot(time_intervals, x[i].flatten(), label="Input Sequence")

            # Plot true and predicted outputs
            ax.scatter(time_intervals, y_true[i], s=100, c='r', label="True")
            ax.scatter(time_intervals, y_pred[i], s=100, c='g', marker='X', label="Predicted")

            ax.set_title(f"Example {i+1}")
            ax.set_xlabel("Time (minutes)")
            ax.set_ylabel("Value")
            ax.legend()

            with self.writer.as_default():
                tf.summary.image(
                    f"Time Series {i}", self.plot_to_image(fig), step=epoch
                )

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        buf = io.BytesIO()

        # Use plt.savefig to save the plot to a PNG in memory.
        plt.savefig(buf, format="png")
        plt.close(figure)
        buf.seek(0)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Expand the dimensions to [1, *, *, 4]
        image = tf.expand_dims(image, 0)

        return image
