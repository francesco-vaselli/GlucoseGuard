
In this project, TensorFlow serves as the foundational library, offering a flexible ecosystem for building and training machine learning models. To establish a comprehensive understanding of the task at hand—time-series forecasting of CGM data—we initially implemented baseline models like Gaussian Processes and a chain of Support Vector Machines, using other dedicated libraries. These traditional techniques provided us with a preliminary metric baseline against which more sophisticated approaches could be compared.

Subsequently, we ventured into the domain of deep learning, employing architectures like Recurrent Neural Networks (RNNs), Convolutional Neural Networks (CNNs), and Attention Networks. These models, known for their ability to capture intricate patterns in data, are elaborated in greater detail in the subsequent sections.

## Comparison of Results

To ascertain the efficacy of our models, we compared our results with those from the reference paper. The key metrics for the binary classification task with a prediction horizon of 30 minutes ([Table 1 of Reference Paper](https://www.nature.com/articles/s41746-021-00480-x/tables/1)) are summarized in the table below. We are unable to compare the loss scores as we use a different dataset.

**We note that our results surpass those of the reference paper and of other notable works in the field**. This is true even for the basic models, which do not leverage data augmentation or transfer learning. We attribute this to the high quantity and good quality of training data at our disposal, showing again the key importance of data when working with ML models.

| Metrics       |  ARIMA       |    GP     |   CNN (Basic)     |   RNN (Basic)    |   Attention (Basic)    | Ref paper (Best)     | Bevan et al[^1] |
|---------------|---------|---------|---------|---------|---------|----------|----------------|
| Accuracy      |         |         |   97.21%      |   **97.44%**      |  97.07%       | 95.98%   | 95.65%         |
| F1 score      |         |         |   79.46%      |   **80.77%**      |  78.52%       | 61.72%   | 57.40%         |
| Sensitivity   |         |         |   **80.44%**      |   80.16%      |  75.38%       | 59.19%   | 49.94%         |
| Precision (PPV)|        |         |   78.51%     |   81.39%      |   **81.94%**      | 67.68%   | 69.00%         |
<!-- | Specificity   |         |         |         |         |         | 98.15%   | 98.61%         |
| NPV           |         |         |         |         |         | 97.55%   | 96.76%         | -->


## Bayesian Hyperparameter Tuning with Keras

To select the best combination of hyperparameters for each model, we leveraged Keras' [BayesianOptimizationOracle](https://keras.io/api/keras_tuner/oracles/bayesian/) for hyperparameter tuning. Unlike traditional methods like grid search or random search, Bayesian optimization provides an intelligent approach to navigating the hyperparameter space. It builds a probabilistic model of the objective function based on past trial data and uses this to predict the most promising hyperparameters to try next. This guided strategy is computationally efficient and often yields more accurate models compared to standard choices.

[^1]: Bevan, R. & Coenen, F. In (eds Bach, K., Bunescu, R., Marling, C. & Wiratunga, N.) Knowledge Discovery in Healthcare Data 2020, Vol. 2675, 100–104 (CEUR Workshop Proceedings, 2020).