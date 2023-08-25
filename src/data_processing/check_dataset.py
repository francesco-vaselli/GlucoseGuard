import numpy as np
import matplotlib.pyplot as plt


def filter_stationary_sequences(train_x, train_y):
    # Define a mask where each entry will be True if the corresponding sample is not stationary
    mask = np.ones(len(train_x), dtype=bool)
    std = np.std(np.concatenate((train_x, train_y), axis=1), axis=1)
    mask[std == 0] = False

    # Apply the mask to filter out stationary sequences
    filtered_train_x = train_x[mask]
    filtered_train_y = train_y[mask]

    return filtered_train_x, filtered_train_y

def filter_stationary_sequences_dataset(ds):

    mask = np.ones(len(ds), dtype=bool)
    std = np.std(ds, axis=1)
    mask[std == 0] = False

    return ds[mask]


data_path = "/home/fvaselli/Documents/PHD/TSA/TSA/data/dataset.npy"
ds = np.load(data_path)
train_x = ds[:, :7]
train_y = ds[:, 7:]
print(train_x.shape, train_y.shape)
# filter out stationary sequences
ds = filter_stationary_sequences_dataset(ds)
train_x = ds[:, :7]
train_y = ds[:, 7:]
# filter_stationary_sequences(train_x, train_y)
print(train_x.shape, train_y.shape)
# restore mean and std
mean = 144.98
std = 58.1

train_x = train_x[2000000:2000000+10, :]
train_y = train_y[2000000:2000000+10, :]
print(train_x.shape, train_y.shape)
train_x = train_x * std + mean
train_y = train_y * std + mean
# plot the data all on the same plot
time_intervals_x = np.arange(0, 5 * train_x[0].shape[0], 5)
time_intervals_y = np.arange(
    5 * train_x[0].shape[0], 5 * train_x[0].shape[0] + 5 * train_y[0].shape[0], 5
)

for i in range(10):
    plt.plot(time_intervals_x, train_x[i], label="Input Sequence")
    plt.scatter(time_intervals_y, train_y[i], s=100, c="r", label="True")
    plt.title(f"Example {0+1}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Value")
    plt.legend()
    plt.show()