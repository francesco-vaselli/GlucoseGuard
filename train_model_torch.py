import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import yaml


class CNNModel(nn.Module):
    def __init__(self, gen_config):
        super(CNNModel, self).__init__()
        config = gen_config["cnn_config"]

        input_channels = 1  # only one feature
        conv_layers = []

        for _ in range(config["n_conv_layers"]):
            conv_layers.append(
                nn.Conv1d(input_channels, config["filters"], config["kernel_size"])
            )
            conv_layers.append(self.get_activation(config["activation"]))
            input_channels = config[
                "filters"
            ]  # subsequent layers expect the same number of channels as the previous layer's filters

        self.convs = nn.Sequential(*conv_layers)
        self.flatten = nn.Flatten()

        # Determine the size after the convolution and flattening
        flattened_size = self.calculate_flattened_size(config)

        dense_layers = []
        for _ in range(config["n_dense_layers"]):
            dense_layers.append(nn.Linear(flattened_size, config["dense_size"]))
            dense_layers.append(self.get_activation(config["activation"]))
            flattened_size = config["dense_size"]

        self.denses = nn.Sequential(*dense_layers)
        self.output = nn.Linear(config["dense_size"], config["output_shape"])

    def calculate_flattened_size(self, config):
        # You will need to calculate the size of the data after it's been through the conv layers
        # This is a simplified example and might need to be adapted based on your specific architecture
        size = 7
        for _ in range(config["n_conv_layers"]):
            size = size - config["kernel_size"] + 1
            size = size // config["stride"] if "stride" in config else size
        return size * config["filters"]

    def get_activation(self, activation_name):
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function: {activation_name}")

    def forward(self, x):
        x = self.convs(x)
        x = self.flatten(x)
        x = self.denses(x)
        x = self.output(x)
        return x


def train(
    data_path, n_train, n_val, n_test, batch_size, epochs, learning_rate, model_config
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # 1. Load the data
    data = np.load(data_path)
    train_x = torch.tensor(data[:n_train, :7], dtype=torch.float32).view(-1, 1, 7)
    train_y = torch.tensor(data[:n_train, 7:], dtype=torch.float32).view(-1, 1, 6)
    val_x = torch.tensor(data[n_train : n_train + n_val, :7], dtype=torch.float32).view(
        -1, 1, 7
    )
    val_y = torch.tensor(data[n_train : n_train + n_val, 7:], dtype=torch.float32).view(
        -1, 1, 6
    )
    test_x = torch.tensor(
        data[n_train + n_val : n_train + n_val + n_test, :7], dtype=torch.float32
    ).view(-1, 1, 7)
    test_y = torch.tensor(
        data[n_train + n_val : n_train + n_val + n_test, 7:], dtype=torch.float32
    ).view(-1, 1, 6)

    # 2. Build the dataset
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 3. Define the CNN model for regression
    model = CNNModel(model_config)
    model.to(device)  # Sending the model to the device

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        print_every_n_batches = 100
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # Forward pass
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if (i + 1) % print_every_n_batches == 0:
                print(
                    f"\tTraining batch {i+1}/{len(train_loader)}, Loss: {loss.item()}"
                )

        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss.append(loss.item())
        avg_val_loss = val_loss_total / len(val_loader)
        print(f"\tValidation Loss: {avg_val_loss}")

    # Plot the loss
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.legend()
    plt.savefig("loss.png")


def main():
    with open("train_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train(
        config["data_path"],
        config["n_train"],
        config["n_val"],
        config["n_test"],
        config["batch_size"],
        config["epochs"],
        config["learning_rate"],
        config["model_config"],
    )


if __name__ == "__main__":
    main()
