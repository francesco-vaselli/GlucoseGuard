
In this project, TensorFlow serves as the foundational library, offering a flexible ecosystem for building and training machine learning models. To establish a comprehensive understanding of the task at hand—time-series forecasting of CGM data—we initially implemented baseline models like Gaussian Processes and a chain of Support Vector Machines, using other dedicated libraries. These traditional techniques provided us with a preliminary metric baseline against which more sophisticated approaches could be compared.

Subsequently, we ventured into the domain of deep learning, employing architectures like Recurrent Neural Networks (RNNs), Convolutional Neural Networks (CNNs), and Attention Networks. These models, known for their ability to capture intricate patterns in data, are elaborated in greater detail in the subsequent sections.

## Comparison of Results

To ascertain the efficacy of our models, we compared our results with those from the reference paper *using data from the six patients in the 2020 test split of the OhioT1DM dataset.*. The key metrics for MAE, RMSE and the binary classification task with a prediction horizon of 30 minutes ([Table 1 of Reference Paper](https://www.nature.com/articles/s41746-021-00480-x/tables/1)) are summarized in the tables below. As our dataset underwent a different normalization process compared to the one used in the reference paper, we extracted the Standard Deviation value directly from the codebase of the reference paper, which was `std = 60.565` as opposed to our `std = 57.941`. We then re-scaled our loss values to provide a fair comparison. We report loss scores computed on both the last 30 minute point of the target sequence and the entire 30 minute sequence. 

|     Model             | 30 min point MAE | 30 min point RMSE  | 30 min seq MAE | 30 min seq RMSE
|------------------|------------|--------------|------------|--------------|
| GP               |21.72 | 37.95| 15.68      | 34.48        |
| SVM Chain        |21.25 | 36.83 |14.56      | 30.12        |
| CNN (Basic)      |**12.50** |**17.88** |**6.54**       | **9.48**         |
| RNN (Basic)      |12.61 | 17.99 |6.64       | 9.59         |
| Attention (Basic)|12.58 | 17.96 |6.62       | 9.57         |
| Ref paper (Best) |13.53      | 19.08        |
| Bevan et al[^1]    |14.37      | 18.23 (18.82a)| | |



| Metrics        | GP    | SVM Chain  | CNN (Basic)  | RNN (Basic)  | Attention (Basic) | Ref paper (Best) | Bevan et al[^1]  |
|----------------|-------|------------|--------------|--------------|-------------------|------------------|------------------|
| Accuracy       | 95.21%| 94.80%     | **96.28%**   | 96.09%       | 96.22%            | 95.98%           | 95.65%           |
| F1             | 67.71%| 59.97%     | **72.24%**   | 70.58%       | 71.61%            | 61.72%           | 57.40%           |
| Sensitivity    | **72.32%**| 56.19% | 69.57%      | 67.50%       | 68.64%            | 59.19%           | 49.94%           |
| Precision      | 63.64%| 64.29%     | **75.11%**   | 73.95%       | 74.86%            | 67.68%           | 69.00%           |
| Specificity    | 96.92%| 97.67%     | **98.28%**   | 98.22%       | **98.28%**        | 98.15%           | 98.61%           |
| NPV            | **97.92%**| 96.77% | 97.74%      | 97.59%       | 97.67%            | 97.55%           | 96.76%           |



*We note that our results surpass those of the reference paper and of other notable works in the field. The performance on the regression task has been improved. More importantly, improvements on Sensitivity, Precision and F1 score are all good indicators that the models are getting better at classifying potentially dangerous hypoglicemic events (the minority class of the dataset).* Note that our models are basic versions, not yet optimized with techniques like data augmentation or transfer learning. We expect performance to escalate upon deploying these improvements. The robustness of simple, baseline models such as GP is evident in their consistent high scores in metrics like Sensitivity. We attribute all of this to the high quantity and good quality of training data at our disposal, showing again the key importance of data when working with ML models. 

However, if instead of evaluating the models on the $\approx 15000$ BG sequences of the OhioT1DM dataset we evaluate them on an independent split of $500000$ sequences from our test dataset, never seen before during training or validation, results are even better than what we obtained for the previous table:


| Metrics        |    GP    |   SVM Chain  |  CNN (Basic) |  RNN (Basic) | Attention (Basic) |
|----------------|----------|--------------|--------------|--------------|-------------------|
| Accuracy       |  89.89%  |    84.47%    |    97.21%    |   **97.44%** |      97.07%       |
| F1 score       |  71.86%  |    52.70%    |    79.46%    |   **80.77%** |      78.52%       |
| Sensitivity    |  72.32%  |    48.46%    |  **80.44%**  |    80.16%    |      75.38%       |
| Precision (PPV)|  71.40%  |    57.74%    |    78.51%    |    81.39%    |    **81.94%**     |
| Specificity    |  93.71%  |    92.29%    |    **98.85%**          |    98.40%         |        98.41%           |
| NPV            |  93.97%  |    89.18%    |      **98.54%**        |    98.53%          |        98.50%           |

This suggests that our models may have picked up unique features in our dataset, which hampers their generalizability across different data sources. However, these metrics, derived from a dataset 33 times larger than the initial one, may be considered more robust estimates of our models capacities. This warrants further investigation into our models' performance.

## Results for data augmentation

Wishing to improve the performance on the minority class (hypoglicemic events), we re-trained our models on an augmented dataset. We used the MixUp data augmentation strategy with $\alpha = 2$ and augmented the minority data from $400,000$ to $1,200,000$. Results on the Ohio test partition for the models trained on this new dataset are reported below:

|     Model          | 30 min point MAE | 30 min point RMSE | 30 min seq MAE | 30 min seq RMSE |
|--------------------|------------------|-------------------|----------------|-----------------|
| CNN (Upsampled)    | 12.30            | 17.71             | 6.43           | 9.37            |
| RNN (Upsampled)    | **12.17**           | **17.54**             | **6.36**           | **9.28**            |
| Attention (Upsampled)| 12.24          | 17.62             | 6.41           | 9.35            |

| Metrics       |  CNN (Upsampled) |  RNN (Upsampled)  | Attention (Upsampled) |
|---------------|----------------|-----------------|---------------------|
| Accuracy      |     95.44%     |     95.63%      |       95.61%        |
| F1 score      |     72.66%     |     73.22%      |       73.38%        |
| Sensitivity   |     87.41% (+17%)     |     86.06% (+19%)     |       87.20%  (+19%)      |
| Precision     |     62.18%  (-13%)   |     63.71%  (-10%)    |       63.34%  (-11%)      |
| Specificity   |     96.04%     |     96.35%      |       96.24%        |
| NPV           |     99.03%     |     98.93%      |       99.02%        |

The + and - are in comparison with the baseline models performance on the Ohio test dataset. We can see that all the models have encountered the so-called "Sensitivity-Precision trade off". Sensitivity measures how well the models identify actual positives, and it improves when we up-sample the minority class. On the other hand, precision gauges the accuracy of the models' positive predictions. When we make the models more eager to predict positives by up-sampling, it sometimes mislabels negatives as positives, thus reducing precision. It's a tug-of-war between catching more true positives (sensitivity) and avoiding false positives (precision). Interestingly, we note a small improvement of the regression loss as well, for all the models.

For addressing this trade off, we could implement weighted loss functions that more heavily penalize false positives, helping to increase precision. Another approach is utilizing ensemble methods, combining models that are strong in either sensitivity or precision to achieve a balanced performance. We could also experiment with adjusting the decision threshold for our classifiera; lowering it could increase precision without severely affecting sensitivity. 
## Bayesian Hyperparameter Tuning with Keras

To select the best combination of hyperparameters for each model, we leveraged Keras' [BayesianOptimizationOracle](https://keras.io/api/keras_tuner/oracles/bayesian/) for hyperparameter tuning. Unlike traditional methods like grid search or random search, Bayesian optimization provides an intelligent approach to navigating the hyperparameter space. It builds a probabilistic model of the objective function based on past trial data and uses this to predict the most promising hyperparameters to try next. This guided strategy is computationally efficient and often yields more accurate models compared to standard choices.

[^1]: Bevan, R. & Coenen, F. In (eds Bach, K., Bunescu, R., Marling, C. & Wiratunga, N.) Knowledge Discovery in Healthcare Data 2020, Vol. 2675, 100–104 (CEUR Workshop Proceedings, 2020).