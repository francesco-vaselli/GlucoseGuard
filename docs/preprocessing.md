Our dataset is composed of multiple .json files per patient. Each patient is identified by a unique ID.
In the following we describe the steps which we have used to extract the raw data from the files, preprocess it and optionally upsampling it.

In the end, we have about a hundred patients to build the train dataset, and we leave out 4 patients to experiment with transfer learning.
The large amount of patients data means we end up with a rather large dataset (5 million time series) when compared to the reference work. 

## Data preprocessing

## Oversampling strategies

## Overview of Codebase Architecture for Patient Data Aggregation and Preprocessing

This codebase comprises a sequence of modular components, each fulfilling a specific role in the pipeline of aggregating and preprocessing Continuous Glucose Monitoring System (CGMS) time-series data. The pipeline has been  developed to enhance usability, modularity, and extensibility. Below is a detailed breakdown of each component.

### Configuration Management via YAML Files

The first component is dedicated to the centralized management of various configurations required for data preprocessing and model training. This is accomplished through a YAML file that contains an organized hierarchy of parameters, such as data directory paths, data scaling options, and smoothing parameters. The adoption of an external YAML configuration file not only enhances the ease of management but also allows for a more flexible system configuration.

### CGMSDataSeg Class: Data Preprocessing

The second core component is the *CGMSDataSeg* class, explicitly designed to handle the segmentation and preprocessing of raw CGMS data. The class is equipped with multiple functionalities like data slicing, scaling, and smoothing. It also offers optional data augmentation techniques, including Gaussian noise and MixUp, to improve the robustness of the resulting dataset. The class thus serves as a comprehensive toolkit for turning raw CGMS time-series data into a refined, machine-learning-ready dataset.
<!-- The *_build_dataset* method in CGMSDataSeg constructs time-series windows from raw glucose readings for machine learning models. Specifically, it takes continuous glucose measurements and slices them into overlapping 'windows' of fixed lengths, defined by sampling_horizon and prediction_horizon. These windows serve as input features (x) and corresponding targets (y) for supervised learning. The method allows for different padding strategies to adjust the shape of the output, catering to the needs of various types of temporal models. -->

### DataReader Utility: Data Collection. 

Our third component is the *DataReader* utility class, which has been created to efficiently read and parse JSON files containing patient-specific time-series data. The utility converts the raw JSON data into Python lists, thus making it far more manageable and ready for subsequent preprocessing stages. It boasts the capability of not just reading, but also smartly interpreting the data based on specific attributes and time intervals.

### Dataset Aggregator: Data Compilation and Transformation

The fourth component is a stand-alone script that functions as a dataset aggregator. Leveraging the DataReader and CGMSDataSeg classes, this script successfully merges data from multiple patients based on their unique identifiers. It is engineered to handle multiple JSON files for each patient and gracefully manage such cases. Post-aggregation, the entire dataset is saved as a NumPy array, making it readily accessible for future machine learning applications.
