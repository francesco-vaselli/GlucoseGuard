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
        # take all json files beginning with {pid}_entries
        file = [   
            os.path.join(data_dir , f)
            for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f))
            and f.startswith(pid + "_entries")
            and f.endswith(".json")
        ]
        # check if file exists
        if len(file) == 0:
            raise ValueError(f"File for patient {pid} not found")
        # check if there are multiple files and handle the case
        if len(file) > 1:
            print(f"Multiple files found for patient {pid}.")

        files += file
        for _ in file:   
            files_ids += [pid]
    
    print(files)
    print(files_ids)
    for f, pid in zip(files, files_ids):
        reader = DataReader(
                "OH", f, 5
            )
        # a patient may have multiple json files
        # so we check if the patient is already in the dict
        if pid not in train_data:
            train_data[pid] = reader.read()
        else:
            train_data[pid] += reader.read()
        
        print(f"Patient {pid} has {len(train_data[pid])} entries.")

    # a dumb dataset instance with first file of data_dir
    train_dataset = CGMSDataSeg("OH", files[0], 5)
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
    ids = config["test_ids"]
    test_ids = config["ids"]
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
    np.save("dataset_99908129_smooth_up.npy", dataset)
    # dataset = tf.data.Dataset.from_tensor_slices((data, targets))
    # save
    # dataset.save("data/dataset")

if __name__ == "__main__":
    main('/home/fvaselli/Documents/PHD/TSA/TSA/configs/data_config.yaml')
