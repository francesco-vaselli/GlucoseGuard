# GlucoseGuard

This code is part of the work done by me for the [Statistical and Machine Learning Models for Time Series Analysis](https://www.sns.it/en/corsoinsegnamento/statistical-and-machine-learning-models-time-series-analysis) PhD Exam.

The reliable prediction of glucose levels in diabetes patients remains a pivotal challenge in biomedical engineering and healthcare informatics. Traditional predictive models have often suffered from inaccuracies and an inability to generalize across different physiological characteristics inherent in patient-specific data. The implications of these shortcomings are nontrivial and bear significant ramifications for clinical decision-making.
In the following, we start from a reference paper and we try to reproduce and expand the work already done, to achieve accurate and robust time series forecasting for blood sugar levels prediction.

The problem is illustrated in the Figure below. We have a series of Blood Glucose Reading at 5 minutes timesteps. We take as input the values of the previous half-hour ("Sampling Horizon") and perform a regression on the following half-hour values ("Prediciton Horizon"). We can also treat the problem as a classification one by looking at where our predictions fall: either above or below the *Hypoglicemia* threshold (80 mg/dL).

![The problem](docs/img/problem.png)

We acknowledge the use of the [OpenAPS Data Commons](https://openaps.org/outcomes/data-commons/) dataset, and we would like to publicly thank the authors and the contributors for the effort of gathering so many real-world patient data and making them accessible.

Part of the codebase is inspired from [AccurateBG](https://github.com/yixiangD/AccurateBG/tree/main) which is released under MIT license
