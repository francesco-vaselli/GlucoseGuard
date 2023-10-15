# aggregate data from multiple patients into one dataset

import yaml
import os

import numpy as np
from CGMSDataSeg import CGMSDataSeg
from data_reader import DataReader


def build_dataset(
    data_dir,
    ids,
    test_ids,
    sampling_horizon,
    prediction_horizon,
    scale,
    outtype,
    smooth,
    target_weight,
    standardize,
    cutpoint,
    augmentation=None,
    augmentation_params=None,
    standardize_by_ref=False,
    standardize_params=None,
):
    # read in all patients data
    train_data = dict()
    files = []
    files_ids = []
    for pid in ids:
        files += [f"/home/fvaselli/Documents/TSA/data/data_ohio/{pid}-ws-testing.xml"]
        reader = DataReader(
            "ohio", f"/home/fvaselli/Documents/TSA/data/data_ohio/{pid}-ws-testing.xml", 5
        )
        train_data[pid] = reader.read()
        
        print(f"Patient {pid} has {len(train_data[pid])} entries.")

    # a dumb dataset instance with first file of data_dir
    train_dataset = CGMSDataSeg("ohio", files[0], 5)
    print(len(train_dataset.data))  # Check length before
    train_pids = set(ids) - set(test_ids)
    local_train_data = []
    for k in train_pids:
        local_train_data += train_data[k]
    train_dataset.data = local_train_data
    print(len(train_dataset.data))  # Check length after
    train_dataset.set_cutpoint = cutpoint
    train_dataset.reset(
        sampling_horizon,
        prediction_horizon,
        scale,
        100,
        smooth,
        outtype,
        target_weight,
        standardize,
    )

    # apply your favorite data augmentation method here
    if augmentation is not None:
        if augmentation == "GaussianNoise":
            train_dataset.gaussian_noise(**augmentation_params)
        elif augmentation == "MixUp":
            train_dataset.mixup(**augmentation_params)
        else:
            raise NotImplementedError(f"{augmentation} not implemented")
        
    data = train_dataset.train_x
    targets = train_dataset.train_y

    if standardize_by_ref:
        print("Standardizing by reference")
        mean = standardize_params["mean"]
        std = standardize_params["std"]
        data = (data - mean) / std
        targets = (targets - mean) / std
    
    return data, targets

def main(data_config):
    # load data config from path
    with open(data_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = config["data_dir"]
    ids = config["ids"]
    test_ids = config["test_ids"]
    sampling_horizon = config["sampling_horizon"]
    prediction_horizon = config["prediction_horizon"]
    scale = config["scale"]
    outtype = config["outtype"]
    smooth = config["smooth"]
    target_weight = config["target_weight"]
    standardize = config["standardize"]
    cutpoint = config["cutpoint"]
    augmentation = config["augmentation"]
    augmentation_params = config["augmentation_params"]
    standardize_by_ref = config["standardize_by_ref"]
    standardize_params = config["standardize_params"]

    data, targets = build_dataset(
        data_dir,
        ids,
        test_ids,
        sampling_horizon,
        prediction_horizon,
        scale,
        outtype,
        smooth,
        target_weight,
        standardize,
        cutpoint,
        augmentation,
        augmentation_params,
        standardize_by_ref,
        standardize_params,
    )

    # save data and targets as numpy arrays, in same file
    dataset = np.concatenate((data, targets), axis=1)
    np.save("/home/fvaselli/Documents/TSA/data/data_ohio/dataset_ohio_smooth_stdbyupsampled.npy", dataset)
    # dataset = tf.data.Dataset.from_tensor_slices((data, targets))
    # save
    # dataset.save("data/dataset")

if __name__ == "__main__":
    main('/home/fvaselli/Documents/TSA/configs/ohio_data_config.yaml')