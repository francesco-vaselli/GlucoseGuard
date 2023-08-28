
import yaml
import os
import argparse
import numpy as np
from CGMSDataSeg import CGMSDataSeg
from data_reader import DataReader


def filter_stationary_sequences_dataset(ds):

    mask = np.ones(len(ds), dtype=bool)
    std = np.std(ds, axis=1)
    mask[std == 0] = False

    return ds[mask]

def upsample_dataset(
    data_path,
    upsampling_strategy,
    upsampling_factor,
    start_file,
    undersample_hyper,
    undersample_ratio,
):
    # 1. Load the data
    ds = np.load(data_path)
    ds = filter_stationary_sequences_dataset(ds)

    # multiply by std and add mean
    ds = ds * 57.940 + 144.982

    dataset = CGMSDataSeg("OH", start_file, 5)
    dataset.data = ds
    dataset.set_cutpoint = -1
    # 2. Upsample the data
    dataset.reset(
        sampling_horizon=7,
        prediction_horizon=6,
        scale=1,
        train_test_ratio=100,
        smooth=False,
        padding="History",
        target_weight=1,
    )

    if undersample_hyper == True:
        dataset.undersampling(ratio=undersample_ratio, padding="History")
        # apply your favorite data augmentation method here
    if upsampling_strategy is not None:
        if upsampling_strategy == "GaussianNoise":
           dataset.gaussian_noise(num_augmentations=upsampling_factor, padding="History", var=3)
        elif upsampling_strategy == "MixUp":
            dataset.mixup(padding="History")
        else:
            raise NotImplementedError(f"{upsampling_strategy} not implemented")
    
    dataset._scale(standardize=True)
    data = dataset.train_x
    targets = dataset.train_y
    return data, targets

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/fvaselli/Documents/PHD/TSA/TSA/data/dataset_smooth.npy",
    )
    parser.add_argument(
        "--upsample_strategy",
        type=str,
        default="MixUp",
    )
    parser.add_argument(
        "--upsample_factor",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--undersample_hyper",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--undersample_ratio",
        type=float,
        default=0.6,
    )

    args = parser.parse_args()
    data_dir = args.dataset
    upsampling_strategy = args.upsample_strategy
    upsampling_factor = args.upsample_factor
    undersample_hyper = args.undersample_hyper
    undersample_ratio = args.undersample_ratio

    data, targets = upsample_dataset(
        data_dir,
        upsampling_strategy,
        upsampling_factor,
        start_file='/home/fvaselli/Documents/PHD/TSA/TSA/data/data_oh/00221634_entries_2018-03-01_to_2018-08-05.json',
        undersample_hyper=undersample_hyper,
        undersample_ratio=undersample_ratio,
    )

    # save data and targets as numpy arrays, in same file
    dataset = np.concatenate((data, targets), axis=1)
    new_name = data_dir.split(".")[0] + "_upsampled.npy"
    np.save(new_name, dataset)

if __name__ == "__main__":
    main()