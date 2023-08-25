import datetime

import matplotlib.pyplot as plt
import numpy as np
from src.data_processing.data_reader import DataReader
from scipy import signal
from sklearn.metrics import mean_squared_error


class CGMSData(object):
    """Data set"""

    def __init__(self, datatype, filepath, sampling_interval):
        self.interval = sampling_interval
        reader = DataReader(datatype, filepath, self.interval)
        self.raw_data = reader.read()
        self.data = list(self.raw_data)
        print(f"Reading {len(self.data)} segments")

        self.sampling_horizon, self.prediction_horizon = 0, 0
        self.scale, self.train_test_ratio = 0, 0
        self.n, self.set_cutpoint = len(self.data), False
        self.train_x, self.train_y, self.train_weights = None, None, None
        self.test_x, self.test_y = None, None
        self.train_n, self.test_n = 0, 0
        self.train_idx = None

    def _smooth(self, window_length, polyorder):
        self.data = list(
            map(
                lambda x: signal.savgol_filter(x, window_length, polyorder),
                filter(lambda x: len(x) > window_length, self.data), # changed from self.raw_data to combine mutpile patients
            )
        )

    def _cut_point(self):
        s = list(map(lambda d: d.size, self.data))
        s = np.cumsum(s)
        if np.isinf(self.train_test_ratio):
            c = s[-1]
        else:
            c = s[-1] * self.train_test_ratio / (1 + self.train_test_ratio)
        return max(np.searchsorted(s, c, side="right"), 1)

    def _build_dataset(self, beg, end, padding):
        print(f"Requesting data from {beg} to {end}")
        x, y = [], []
        l = self.sampling_horizon + self.prediction_horizon
        for d in self.data[beg:end]:
            d = np.array(d)
            for i in range(
                d.size - self.sampling_horizon - self.prediction_horizon + 1
            ):
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

    def _scale(self, standardize):
        if standardize:
            # Determine if test data exists
            test_exists = len(self.test_x) > 0 and len(self.test_y) > 0

            # Calculate mean and std based on the presence of test data
            if test_exists:
                all_data = np.concatenate(
                    [
                        np.concatenate(self.train_x, axis=0),
                        np.concatenate(self.train_y, axis=0),
                        np.concatenate(self.test_x, axis=0),
                        np.concatenate(self.test_y, axis=0),
                    ],
                    axis=0,
                )
            else:
                all_data = np.concatenate(
                    [
                        np.concatenate(self.train_x, axis=0),
                        np.concatenate(self.train_y, axis=0),
                    ],
                    axis=0,
                )

            mean = np.mean(all_data)
            std = np.std(all_data)

            # Log the values
            print(f"Computed Mean: {mean}, Computed Std: {std}")

            # Apply standardization
            self.train_x = (self.train_x - mean) / std
            self.train_y = (self.train_y - mean) / std

            if test_exists:
                self.test_x = (self.test_x - mean) / std
                self.test_y = (self.test_y - mean) / std
        else:
            self.train_x *= self.scale
            self.train_y *= self.scale
            if len(self.test_x) > 0 and len(self.test_y) > 0:
                self.test_x *= self.scale
                self.test_y *= self.scale


    def reset(
        self,
        sampling_horizon,
        prediction_horizon,
        scale,
        train_test_ratio,
        smooth,
        padding,
        target_weight,
    ):
        self.sampling_horizon = sampling_horizon
        self.prediction_horizon = prediction_horizon
        self.scale = scale
        self.train_test_ratio = train_test_ratio

        if smooth:
            window_length = sampling_horizon
            if window_length % 2 == 0:
                window_length += 1
            self._smooth(window_length, window_length - 4)
        print("# time series: {}".format(len(self.data)))
        c = self._cut_point()
        self.train_x, self.train_y = self._build_dataset(0, c, padding)
        self.test_x, self.test_y = self._build_dataset(c, len(self.data), padding)
        self.train_n = self.train_x.shape[0]
        self.test_n = self.test_x.shape[0]
        print("Train data size: %d" % self.train_n)
        print("Test data size: %d" % self.test_n)
        self._scale(True)

        self.train_weights = None
        if padding != "None":
            l = self.train_y.shape[1]
            self.train_weights = np.full(l, (1 - target_weight) / (l - 1))
            self.train_weights[-1] = target_weight

        self.train_idx = np.random.permutation(self.train_n)

    def t0_baseline(self):
        y = self.test_y
        if y.ndim == 2:
            y = y[:, -1]
        return mean_squared_error(y, self.test_x[:, -1]) ** 0.5 / self.scale

    def train_next_batch(self, batch_size):
        if self.train_idx.size < batch_size:
            self.train_idx = np.random.permutation(self.train_n)
        idx = self.train_idx[:batch_size]
        self.train_idx = self.train_idx[batch_size:]
        return self.train_x[idx], self.train_y[idx], self.train_weights

    def test(self):
        weights = None
        if self.train_weights is not None:
            weights = np.zeros_like(self.train_y[0])
            weights[-1] = 1
        return self.test_x, self.test_y, weights

    def render_data(self, n=3):
        plt.figure()
        for d in self.data[:n]:
            plt.plot(d, marker="o")
        plt.xlabel("Time (%d min)" % (self.interval))

        dist_l2 = 0.04
        x0 = None
        n = 0
        for x in self.train_x:
            if np.var(x) < 0.001:
                continue
            l = np.linalg.norm(self.train_x - x, axis=1)
            idx = np.nonzero(l < dist_l2)[0]
            if idx.size > n:
                n = idx.size
                x0 = x
        l = np.linalg.norm(self.train_x - x0, axis=1)
        idx = np.nonzero(l < dist_l2)[0]
        plt.figure()
        for i in idx:
            plt.plot(self.train_x[i] / self.scale, marker="o")
            # now plot the corresponding y in a way that makes it clear
            # which x it corresponds to
            y = self.train_y[i]
            plt.plot(
                np.arange(self.sampling_horizon, self.sampling_horizon + y.size),
                y / self.scale,
                marker="o",
            )

        plt.xlabel("Time (%d min)" % (self.interval))
        plt.title("%d samples" % n)
        plt.show()

    def test_patient(self, ptid=-1):
        x = []
        while self.data[ptid].size < (self.sampling_horizon + self.prediction_horizon):
            ptid -= 1
        d = self.data[ptid]
        for i in range(d.size - self.sampling_horizon - self.prediction_horizon + 1):
            x.append(d[i : (i + self.sampling_horizon)])
        return ptid, np.array(x) * self.scale

    def render_prediction(self, ptid, y, yerr=None, show=True):
        plt.figure()
        plt.plot(self.data[ptid], "bo-", label="Truth")
        x = np.arange(y.size) + (self.sampling_horizon + self.prediction_horizon - 1)
        if yerr is not None:
            plt.errorbar(
                x, y / self.scale, yerr=yerr / self.scale, fmt="none", ecolor="grey"
            )
        plt.plot(x, y / self.scale, "gv-", label="Prediction")
        plt.legend(loc="best")
        plt.xlabel("Time (%d min)" % (self.interval))
        if show:
            plt.show()
        else:
            plt.savefig("prediction_%d.png" % (ptid + len(self.data)))


def main():
    data = CGMSData(
        "OH",
        "data/04762925/direct-sharing-31/04762925_entries_2016-01-01_to_2018-04-17.json",
        5,
    )

    data.reset(7, 6, 1, 3, True, "History", 0.5)
    data.render_data()


if __name__ == "__main__":
    main()
