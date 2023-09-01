# Data Preprocessing

## Inaccuracies in Current CGM Technologies

Continuous Glucose Monitoring (CGM) technologies have revolutionized the management of diabetes by providing real-time, high-frequency glucose readings. However, these systems are not without their flaws. For example, sensor drift and noise can introduce inconsistencies in the data. Calibration errors can also skew measurements, making it difficult to rely solely on CGM data for clinical decisions. Additionally, factors like temperature, humidity, and body movement can interfere with the sensor's performance. This variability necessitates robust preprocessing steps to improve the reliability of CGM data.

## Savitzky-Golay Filter for Data Smoothing

One widely-adopted method to address these inaccuracies is the application of the [Savitzky-Golay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) filter. This polynomial-based smoothing technique is particularly well-suited for time-series data with noise, as it allows for the preservation of high-frequency features while reducing random fluctuations. In essence, it fits a polynomial of a certain degree to a window of adjacent data points, calculating a smoothed value for each point in the series. This proves invaluable for CGM data, where capturing the nuances of glucose changes without the noise is critical for effective forecasting.
We applied the filter following the example of our reference paper.

## Standardization of ML Inputs

Lastly, it is a standard practice in machine learning to normalize the features to bring them onto a similar scale. For this project, the CGM data was preprocessed by subtracting the mean and dividing by the standard deviation. This process, known as z-score normalization or standardization, ensures that the data has zero mean and a standard deviation of one. This normalization enhances the performance and stability of machine learning algorithms, particularly those sensitive to the scale of input features, such as gradient-based models.