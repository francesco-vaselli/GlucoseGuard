import matplotlib.pyplot as plt
import numpy as np
from CGMSdata import CGMSData
from MixUp import MixUp as mixup


class CGMSDataSeg(CGMSData):
    """Data set"""

    def __init__(self, datatype, filepath, sampling_interval):
        super().__init__(datatype, filepath, sampling_interval)
        self._feature = None
        self._hypo_th = 80
        self._border_th = 10
        self._hypo_train_x = None
        self._hypo_train_y = None
        self.border_train_x = None
        self.border_train_y = None
        self._nonhypo_train_x = None
        self._nonhypo_train_y = None
        self._original_train_x = None
        self._original_train_y = None
        self.gan_data = None
        self.alpha = 0.4

    def _build_dataset(self, beg, end, padding):
        print(f"Building dataset, requesting data from {beg} to {end}")
        x, y = [], []
        l = self.sampling_horizon + self.prediction_horizon
        for j in range(beg, end):
            d = np.array(self.data[j])
            for i in range(
                d.size - self.sampling_horizon - self.prediction_horizon + 1
            ):
                if self.feature is not None:
                    x.append(
                        np.hstack((self.feature[j], d[i : (i + self.sampling_horizon)]))
                    )
                else:
                    x.append(d[i : (i + self.sampling_horizon)])
                if padding == "History":
                    y.append(d[(i + self.sampling_horizon) : (i + l)])
                else:
                    y.append(d[i + l - 1])
        if padding == "None" or padding == "History":
            return np.array(x), np.array(y)
        if padding == "Same":
            return np.array(x), np.tile(y, [self.sampling_horizon, 1]).T
        raise ValueError("Unsupported padding " + padding)

    @property
    def feature(self):
        return self._feature

    @feature.setter
    def feature(self, value):
        if len(value) != len(self.data):
            print("Feature and data length mismatch, set to None")
            self._feature = None
        else:
            self._feature = value

    def reset(
        self,
        sampling_horizon,
        prediction_horizon,
        scale,
        train_test_ratio,
        smooth,
        padding,
        target_weight,
        standardize=False,
    ):
        self.sampling_horizon = sampling_horizon
        self.prediction_horizon = prediction_horizon
        self.scale = scale
        self.train_test_ratio = train_test_ratio
        print("Resetting data...")
        if smooth:
            window_length = sampling_horizon
            if window_length % 2 == 0:
                window_length += 1
            self._smooth(window_length, window_length - 4)
        x, y = self._build_dataset(0, len(self.data), padding)

        train_ratio = self.train_test_ratio / (1 + self.train_test_ratio)
        c = int(x.shape[0] * train_ratio)
        if self.set_cutpoint:
            if self.set_cutpoint < x.shape[0] and self.set_cutpoint > 0:
                c = self.set_cutpoint
            elif self.set_cutpoint < 0:
                print("Train data requested beyond limit, using all as train data")
                c = x.shape[0]
        self._original_train_x = x[0:c]
        self._original_train_y = y[0:c]
        # cast as float32, some strange error when processing the whole dataset
        self._original_train_x = self._original_train_x.astype(np.float32)
        self._original_train_y = self._original_train_y.astype(np.float32)
        # detect hypo in training data
        print("Shape of original train data: ", self._original_train_y.shape)
        if padding == "Same":
            hypo_loc = np.where(self._original_train_y[:, 0] < self._hypo_th)
            border_loc = np.where(
                np.abs(self._original_train_y[:, 0] - self._hypo_th) < self._border_th
            )
            nonhypo_loc = np.where(self._original_train_y[:, 0] >= self._hypo_th)
        elif padding == "None":
            hypo_loc = np.where(self._original_train_y < self._hypo_th)
            border_loc = np.where(
                np.abs(self._original_train_y - self._hypo_th) < self._border_th
            )
            nonhypo_loc = np.where(self._original_train_y >= self._hypo_th)
        elif padding == "History":
            print(type(self._original_train_y[:, -1]), type(self._hypo_th))
            print(self._original_train_y[:, -1].shape)
            print(self._original_train_y[:, -1])
            print(self._hypo_th)
            print(np.any(np.isnan(self._original_train_y)))
            print(np.any(np.isinf(self._original_train_y)))

            hypo_loc = np.where(self._original_train_y[:, -1] < self._hypo_th)
            border_loc = np.where(
                np.abs(self._original_train_y[:, -1] - self._hypo_th) < self._border_th
            )
            nonhypo_loc = np.where(self._original_train_y[:, -1] >= self._hypo_th)
        hypo_loc = hypo_loc[0]
        border_loc = border_loc[0]
        nonhypo_loc = nonhypo_loc[0]
        self._hypo_train_x = self._original_train_x[hypo_loc]
        self._hypo_train_y = self._original_train_y[hypo_loc]
        self.border_train_x = self._original_train_x[border_loc]
        self.border_train_y = self._original_train_y[border_loc]
        self._nonhypo_train_x = self._original_train_x[nonhypo_loc]
        self._nonhypo_train_y = self._original_train_y[nonhypo_loc]
        self.train_x = np.copy(self._original_train_x)
        self.train_y = np.copy(self._original_train_y)
        self.test_x = x[c:]
        self.test_y = y[c:]
        self.test_n = self.test_x.shape[0]
        print("#" * 28 + " Data structure summary " + "#" * 28)
        print("Hypo/no_hypo: {}/{}".format(len(hypo_loc), len(nonhypo_loc)))
        print(
            "Found {} continuous time series, however not all will pass the window requirements".format(
                len(self.data)
            )
        )
        print(
            "Data shape: {}, Train/test: {}/{}".format(
                x.shape, self.train_x.shape[0], self.test_n
            )
        )
        if self.test_n != 0:
            print(
                "Train test ratio: {:.2f}".format(self.train_x.shape[0] / self.test_n)
            )
        else:
            print("Test data is empty...")
        print("#" * 80)
        self._scale(standardize)

        self.train_weights = None
        if padding != "None":
            l = self._original_train_y.shape[1]
            self.train_weights = np.full(l, (1 - target_weight) / (l - 1))
            self.train_weights[-1] = target_weight
        self.train_n = self.train_x.shape[0]
        self.train_idx = np.random.permutation(self.train_n)
        # self.gan_data = np.loadtxt("../time-gan/results/gen_hypo.txt")

    def undersampling(self, ratio, padding):
        if padding == "Same":
            nonhypo_loc = np.where(self._original_train_y[:, 0] >= self._hypo_th)
        elif padding == "None":
            nonhypo_loc = np.where(self._original_train_y >= self._hypo_th)
        nonhypo_loc = nonhypo_loc[0]
        nonhypo_train_x = self._original_train_x[nonhypo_loc]
        nonhypo_train_y = self._original_train_y[nonhypo_loc]
        num = int(ratio * nonhypo_loc.size)
        indx = np.random.choice(nonhypo_loc.size, num)
        self.train_x = np.vstack((nonhypo_train_x[indx], self._hypo_train_x))
        if padding == "Same":
            self.train_y = np.vstack((nonhypo_train_y[indx], self._hypo_train_y))
        elif padding == "None":
            self.train_y = np.hstack((nonhypo_train_y[indx], self._hypo_train_y))
        self.train_x *= self.scale
        self.train_y *= self.scale
        self.train_n = self.train_x.shape[0]
        print("After {} undersampling, {} train data".format(ratio, self.train_n))
        self.train_idx = np.arange(self.train_n)

    def mixup(self, padding):
        """
        generate mixup training data
        """
        m = 2
        alpha = 2
        model = mixup(
            self._original_train_x,
            self._original_train_y,
            self._hypo_train_x,
            self._hypo_train_y,
            self._nonhypo_train_x,
            self._nonhypo_train_y,
            alpha,
            m,
        )
        option = "unbiased"
        option = "minority"
        # option = 'inner'
        new_train_x, new_train_y = model.mixup_by(option)
        self.train_x = np.vstack((self._original_train_x, new_train_x))
        if padding == "Same" or padding == "History":
            self.train_y = np.vstack((self._original_train_y, new_train_y))
        elif padding == "None":
            self.train_y = np.hstack((self._original_train_y, np.squeeze(new_train_y)))
        self.train_x *= self.scale
        self.train_y *= self.scale
        self.train_n = self.train_x.shape[0]
        print("After {} fold mixing up, {} train data".format(m, self.train_n))
        self.train_idx = np.arange(self.train_n)

    def gaussian_noise(self, num_augmentations=1, padding="Same"):
        """
        Apply Gaussian noise to the input data to augment it.

        :param num_augmentations: How many times to apply different Gaussian noise to the same input sequences.
        :param padding: Padding method for the output sequences.
        """

        # Define variance for Gaussian noise
        var = 0.1

        augmented_train_x = [self._original_train_x]
        augmented_train_y = [self._original_train_y]

        for _ in range(num_augmentations):
            # Generate Gaussian noise
            gaussian_noise = np.random.normal(0, var, size=self._hypo_train_x.shape)

            # Apply Gaussian noise to the input sequences
            augmented_train_x.append(self._hypo_train_x + gaussian_noise)
            augmented_train_y.append(self._hypo_train_y)

        # Concatenate original and augmented data along the first axis (rows)
        self.train_x = np.vstack(augmented_train_x)

        if padding == "Same" or padding == "History":
            self.train_y = np.vstack(augmented_train_y)
        elif padding == "None":
            self.train_y = np.hstack(
                (self._original_train_y, np.squeeze(np.vstack(augmented_train_y)))
            )
        else:
            raise ValueError("Unsupported padding method: " + padding)

        # Scale data
        self.train_x *= self.scale
        self.train_y *= self.scale

        self.train_n = self.train_x.shape[0]
        self.train_idx = np.arange(self.train_n)

        print("After adding gaussian noise, {} train data".format(self.train_n))

    def gan_generator(self, padding="Same"):
        idx = np.random.randint(0, self.gan_data.shape[0], self._hypo_train_x.shape[0])
        new_train_x = self.gan_data[idx, : self.sampling_horizon]
        new_train_y = np.tile(
            self.gan_data[idx, -1][:, None], (1, self.sampling_horizon)
        )

        self.train_x = np.vstack((self._original_train_x, new_train_x))
        if padding == "Same":
            self.train_y = np.vstack((self._original_train_y, new_train_y))
        elif padding == "None":
            self.train_y = np.hstack((self._original_train_y, np.squeeze(new_train_y)))
        self.train_x *= self.scale
        self.train_y *= self.scale
        self.train_n = self.train_x.shape[0]
        self.train_idx = np.arange(self.train_n)
        print("after GAN generator, {} train data".format(self.train_n))

    def plot_hist(self, padding):
        plt.figure()
        if padding == "Same":
            plt.hist(self.train_y[:, 0])
        elif padding == "None":
            plt.hist(self.train_y)
        plt.vlines(self._hypo_th * self.scale, 0, 20000, "r")
        plt.xlabel("Scaled BG")
        plt.ylabel("Data Count")
        plt.show()


def main():
    data = CGMSDataSeg(
        "OH",
        "data/04762925/direct-sharing-31/04762925_entries_2016-01-01_to_2018-04-17.json",
        5,
    )

    data.reset(7, 6, 1, 3, True, "History", 0.5)
    # data.gaussian_noise(num_augmentations=3, padding="History")
    data.mixup(padding="History")
    data.render_data()


if __name__ == "__main__":
    main()
