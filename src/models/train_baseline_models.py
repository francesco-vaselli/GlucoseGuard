import numpy as np
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import itertools
import yaml
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils import filter_stationary_sequences_dataset, check_dataset_correlations
import pickle
from sklearn.svm import SVR


def load_data(data_path, n_train, n_val, n_test):
    ds = np.load(data_path)
    ds = filter_stationary_sequences_dataset(ds)

    # shuffle
    np.random.shuffle(ds)

    train_x = ds[:n_train, :7]
    train_y = ds[:n_train, 7:]
    val_x = ds[n_train : n_train + n_val, :7]
    val_y = ds[n_train : n_train + n_val, 7:]
    test_x = ds[n_train + n_val : n_train + n_val + n_test, :7]
    test_y = ds[n_train + n_val : n_train + n_val + n_test, 7:]
    return train_x, train_y, val_x, val_y, test_x, test_y


def train_evaluate_arima(
    train_x, train_y, val_x, val_y, test_x, test_y, order=(5, 1, 0)
):
    # Train ARIMA on targets (or some transformed version of the data)
    model = ARIMA(
        train_y, order=order, exog=train_x
    )  # note the `exog` parameter for including additional features
    model_fit = model.fit()
    pickle.dump(model_fit, open("saved_models/arima_model.pkl", "wb"))

    # Validate the model
    predictions_val = model_fit.forecast(steps=len(val_y), exog=val_x)
    mse_val = mean_squared_error(val_y, predictions_val)
    print(f"Validation MSE for ARIMA model: {mse_val}")

    # Test the model
    predictions_test = model_fit.forecast(steps=len(test_y), exog=test_x)
    mse_test = mean_squared_error(test_y, predictions_test)
    print(f"Test MSE for ARIMA model: {mse_test}")
    return predictions_test


def train_evaluate_gp(train_x, train_y, val_x, val_y, test_x, test_y, config):
    constant_value = config["gp"]["kernel"]["constant"]
    constant_bounds = tuple(config["gp"]["kernel"]["constant_bounds"])
    rbf_value = config["gp"]["kernel"]["rbf"]
    rbf_bounds = tuple(config["gp"]["kernel"]["rbf_bounds"])
    n_restarts_optimizer = config["gp"]["n_restarts_optimizer"]

    kernel = C(constant_value, constant_bounds) * RBF(rbf_value, rbf_bounds)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts_optimizer
    )

    # Fit GP model using training features and targets
    gp.fit(train_x, train_y)
    pickle.dump(gp, open("saved_models/gp_model.pkl", "wb"))

    # Validate the model
    y_pred_mean, y_pred_std = gp.predict(val_x, return_std=True)
    mse_val = mean_absolute_error(val_y, y_pred_mean)
    print(f"Validation MAE for GP model: {mse_val}")

    # Test the model
    y_pred_mean_test, y_pred_std_test = gp.predict(test_x, return_std=True)
    mse_test = mean_absolute_error(test_y, y_pred_mean_test)
    print(f"Test MAE for GP model: {mse_test}")

    return y_pred_mean_test

def train_evaluate_svm(train_x, train_y, val_x, val_y, test_x, test_y, config):
    # Configure the SVR model
    kernel = config["svm"]["kernel"]
    C = config["svm"]["C"]
    gamma = config["svm"]["gamma"]
    svm_model = SVR(kernel=kernel, C=C, gamma=gamma)
    
    # Fit the SVR model
    svm_model.fit(train_x, train_y.ravel())  # ravel() is used to flatten y for fitting
    
    # Save the model
    pickle.dump(svm_model, open("saved_models/svm_model.pkl", "wb"))

    # Validate the model
    y_pred_val = svm_model.predict(val_x)
    mse_val = mean_absolute_error(val_y, y_pred_val)
    print(f"Validation MAE for SVM model: {mse_val}")

    # Test the model
    y_pred_test = svm_model.predict(test_x)
    mse_test = mean_absolute_error(test_y, y_pred_test)
    print(f"Test MAE for SVM model: {mse_test}")

    return y_pred_test

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
        # fig, ax = plt.subplots(figsize=(12, 6))

        # # Plot input time series
        # ax.plot(time_intervals_x, x[i], label="Input Sequence")

        # # Plot true and predicted outputs
        # ax.scatter(time_intervals_y, y_true[i], s=100, c="r", label="True")
        # ax.scatter(
        #     time_intervals_y, y_pred[i], s=100, c="g", marker="X", label="Predicted"
        # )

        # ax.set_title(f"Example {i+1}")
        # ax.set_xlabel("Time (minutes)")
        # ax.set_ylabel("Value")
        # ax.legend()
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

        # save
        fig.savefig(f"{save_path}{title}_{i}.png")


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# Load YAML Config
config = load_config("configs/train_baselines_config.yaml")


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


if __name__ == "__main__":
    # Use parameters from YAML config
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(
        config["data"]["path"],
        config["data"]["n_train"],
        config["data"]["n_val"],
        config["data"]["n_test"],
    )
    # check_dataset_correlations(train_x, train_y, "baseline_figures/")

    # print("------ ARIMA ------")
    # true_label_arima = test_y
    # pred_label_arima = train_evaluate_arima(
    #     train_x,
    #     train_y,
    #     val_x,
    #     val_y,
    #     test_x,
    #     test_y,
    #     order=(config["arima"]["p"], config["arima"]["d"], config["arima"]["q"]),
    # )
    # # save arima model


    # plot_beautiful_fig(
    #     test_x[:3],
    #     true_label_arima[:3],
    #     pred_label_arima[:3],
    #     "arima_figs",
    #     "baseline_figures/",
    #     config["data"]["mean"],
    #     config["data"]["std"],
    # )
    # # Further Evaluation for ARIMA
    # (
    #     fpr,
    #     tpr,
    #     roc_auc,
    #     accuracy,
    #     sensitivity,
    #     specificity,
    #     precision,
    #     npv,
    #     f1,
    # ) = check_classification(
    #     true_label_arima,
    #     pred_label_arima,
    #     threshold=config["metrics"]["threshold"],
    #     std=config["data"]["std"],
    #     mean=config["data"]["mean"],
    # )
    # cm = confusion_matrix(true_label_arima, pred_label_arima)
    # cm_fig = plot_confusion_matrix(cm, class_names=["Hyper", "Hypo"])
    # # save cm_fig
    # cm_fig.savefig("baseline_figures/confusion_matrix.png")
    # # save metrics to .txt file
    # with open("baseline_figures/metrics_arima.txt", "w") as f:
    #     f.write(
    #         f"Accuracy: {accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\nPrecision: {precision}\nNPV: {npv}\nF1: {f1}"
    #     )

    print("------ Gaussian Process ------")
    true_label_gp = test_y
    pred_label_gp = train_evaluate_gp(train_x, train_y, val_x, val_y, test_x, test_y, config)

    plot_beautiful_fig(
        test_x[:3],
        test_y[:3],
        pred_label_gp[:3],
        "gp_figs",
        "baseline_figures/",
        config["data"]["mean"],
        config["data"]["std"],
    )
    # save gp model

    # Further Evaluation for GP
    # Similar to what we did for ARIMA
    (
        true_label_gp,
        pred_label_gp,
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
        true_label_gp,
        pred_label_gp,
        threshold=config["metrics"]["threshold"],
        std=config["data"]["std"],
        mean=config["data"]["mean"],
    )
    cm = confusion_matrix(true_label_gp, pred_label_gp)
    cm_fig = plot_confusion_matrix(cm, class_names=["Hyper", "Hypo"])
    # save cm_fig
    plt.savefig("baseline_figures/confusion_matrix_gp.png")
    # save metrics to .txt file
    with open("baseline_figures/metrics_gp.txt", "w") as f:
        f.write(
            f"Accuracy: {accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\nPrecision: {precision}\nNPV: {npv}\nF1: {f1}"
        )

    print("------ Support Vector Machine ------")
    true_label_svm = test_y
    pred_label_svm = train_evaluate_svm(train_x, train_y, val_x, val_y, test_x, test_y, config)

    # Use your existing function to plot results
    plot_beautiful_fig(
        test_x[:3],
        test_y[:3],
        pred_label_svm[:3],
        "svm_figs",
        "baseline_figures/",
        config["data"]["mean"],
        config["data"]["std"],
    )

    # Further Evaluation for SVM
    # Similar to what we did for ARIMA
    (
        true_label_svm,
        pred_label_svm,
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
        true_label_svm,
        pred_label_svm,
        threshold=config["metrics"]["threshold"],
        std=config["data"]["std"],
        mean=config["data"]["mean"],
    )
    cm = confusion_matrix(true_label_svm, pred_label_svm)
    cm_fig = plot_confusion_matrix(cm, class_names=["Hyper", "Hypo"])
    # save cm_fig
    plt.savefig("baseline_figures/confusion_matrix_svm.png")
    # save metrics to .txt file
    with open("baseline_figures/metrics_svm.txt", "w") as f:
        f.write(
            f"Accuracy: {accuracy}\nSensitivity: {sensitivity}\nSpecificity: {specificity}\nPrecision: {precision}\nNPV: {npv}\nF1: {f1}"
        )
