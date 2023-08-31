Our dataset is composed of multiple .json files per patient. Each patient is identified by a unique ID.
In the following we describe the steps which we have used to extract the raw data from the files, preprocess it and optionally upsampling it.

In the end, we have about a hundred patients to build the train dataset, and we leave out 4 patients to experiment with transfer learning.
The large amount of patients data means we end up with a rather large dataset (5 million time series) when compared to the reference work. 