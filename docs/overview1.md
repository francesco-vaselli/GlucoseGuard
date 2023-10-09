
In this project, TensorFlow serves as the foundational library, offering a flexible ecosystem for building and training machine learning models. To establish a comprehensive understanding of the task at hand—time-series forecasting of CGM data—we initially implemented baseline models like Gaussian Processes and a chain of Support Vector Machines, using other dedicated libraries. These traditional techniques provided us with a preliminary metric baseline against which more sophisticated approaches could be compared.

Subsequently, we ventured into the domain of deep learning, employing architectures like Recurrent Neural Networks (RNNs), Convolutional Neural Networks (CNNs), and Attention Networks. These models, known for their ability to capture intricate patterns in data, are elaborated in greater detail in the subsequent sections.

## Comparison of Results

To ascertain the efficacy of our models, we compared our results with those from the reference paper *on the 6 patients form the 2020 test split of the OhioT1DM dataset*. The key metrics for the binary classification task with a prediction horizon of 30 minutes ([Table 1 of Reference Paper](https://www.nature.com/articles/s41746-021-00480-x/tables/1)) are summarized in the table below. 

**TODO** add losses comparison

| Metrics        | GP    | SVM   | CNN   | RNN   | Attention | Ref paper (Best) | Bevan et al[^1]  |
|----------------|-------|-------|-------|-------|-----------|------------------|------------------|
| Accuracy       | 95.21%| 94.80%|**96.28%**| 96.09%| 96.22%    | 95.98%           | 95.65%           |
| Sensitivity    | **72.32%**| 56.19%| 69.57%| 67.50%| 68.64%    | 59.19%           | 49.94%           |
| Specificity    | 96.92%| 97.67%| **98.28%**| 98.22%| **98.28%**    | 98.15%           | 98.61%           |
| Precision      | 63.64%| 64.29%| **75.11%**| 73.95%| 74.86%    | 67.68%           | 69.00%           |
| NPV            | **97.92%**| 96.77%| 97.74%| 97.59%| 97.67%    | 97.55%           | 96.76%           |
| F1             | 67.71%| 59.97%| **72.24%**| 70.58%| 71.61%    | 61.72%           | 57.40%           |


*We note that our results surpass those of the reference paper and of other notable works in the field*. It should be noted that this are basic models, which do not leverage data augmentation or transfer learning. Performance is expected to increase when deploying these improvements. The robustenss of simple, baseline models such as GP can be appreciated seeing how they retain high scores on, e.g., Sensitivity. We attribute all of this to the high quantity and good quality of training data at our disposal, showing again the key importance of data when working with ML models. 

However, if instead of evaluating the models on the $\approx 15000$ BG sequences of the OhioT1DM dataset we evaluate them on an indipendent split of 500000 sequences from our test dataset, never before seeing during training, results are better than what we obtained for the previous table:


| Metrics        |    GP    |   SVM Chain  |  CNN (Basic) |  RNN (Basic) | Attention (Basic) |
|----------------|----------|--------------|--------------|--------------|-------------------|
| Accuracy       |  89.89%  |    84.47%    |    97.21%    |   **97.44%** |      97.07%       |
| F1 score       |  71.86%  |    52.70%    |    79.46%    |   **80.77%** |      78.52%       |
| Sensitivity    |  72.32%  |    48.46%    |  **80.44%**  |    80.16%    |      75.38%       |
| Precision (PPV)|  71.40%  |    57.74%    |    78.51%    |    81.39%    |    **81.94%**     |
| Specificity    |  93.71%  |    92.29%    |              |              |                   |
| NPV            |  93.97%  |    89.18%    |              |              |                   |

This may be indicating that our models are learning some intrinsic feature of our dataset, which makes generalization on different data sources more difficult. This issue has yet to be investigated further.

## Bayesian Hyperparameter Tuning with Keras

To select the best combination of hyperparameters for each model, we leveraged Keras' [BayesianOptimizationOracle](https://keras.io/api/keras_tuner/oracles/bayesian/) for hyperparameter tuning. Unlike traditional methods like grid search or random search, Bayesian optimization provides an intelligent approach to navigating the hyperparameter space. It builds a probabilistic model of the objective function based on past trial data and uses this to predict the most promising hyperparameters to try next. This guided strategy is computationally efficient and often yields more accurate models compared to standard choices.

[^1]: Bevan, R. & Coenen, F. In (eds Bach, K., Bunescu, R., Marling, C. & Wiratunga, N.) Knowledge Discovery in Healthcare Data 2020, Vol. 2675, 100–104 (CEUR Workshop Proceedings, 2020).