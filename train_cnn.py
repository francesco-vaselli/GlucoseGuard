import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense

def train_cnn():
    # 1. Load the data
    train_x = np.load('./data/dataset.npy')[0]
    train_y = np.load('./data/dataset.npy')[1]

    # Ensure that train_x has the right shape [samples, timesteps, features]
    if len(train_x.shape) == 2:
        train_x = np.expand_dims(train_x, axis=-1)

    # load yaml config 

    # 2. Build the dataset
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000  # For shuffling the dataset

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # 3. Define the 1D CNN model for regression
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=train_x.shape[1:]),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(train_y.shape[1])  # Output layer for regression, so no activation
    ])

    model.compile(optimizer='adam', loss='mse')  # Using Mean Squared Error for regression tasks

    # Training the model
    EPOCHS = 10
    model.fit(train_dataset, epochs=EPOCHS)

def main():
    # load cnn config from yaml
    with open('cnn_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    batch_size = config['batch_size']
    buffer_size = config['buffer_size']
    epochs = config['epochs']
    optimizer = config['optimizer']
    loss = config['loss']
    learning_rate = config['learning_rate']
    filter_shape = config['filter_shape']
    kernel_size = config['kernel_size']
    train_cnn()