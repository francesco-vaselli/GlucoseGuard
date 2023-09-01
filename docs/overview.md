
Our dataset is composed of multiple .json files per patient. Each patient is identified by a unique ID.
In the following sections we describe the steps which we have used to extract the raw data from the files, preprocess it and optionally perform data augmentation.

In the end, we have about a hundred patients to build the train dataset, and we leave out 4 patients to experiment with transfer learning.
The large amount of patients data means we end up with a rather large dataset (5 million time series) when compared to the reference work. Each time series is composed by 13 points, each 5 minutes apart: 7 inputs and 6 targets.
Training is performed on 3 million time series, validation on 300000 and testing on a separate split of 500000. The data comes from different patients and is shuffled before training.

We produced several versions of our dataset, with and without data augmentation. You can find more about that in the relevant section.