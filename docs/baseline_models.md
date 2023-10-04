## Models descriptions

We implemented two baseline models: a Gaussian Process regressor and a *regressor chain* of Support Vector Machine regressors.
Both classes of models are already quite complex and include several hyperparameters to tune. 

- Gaussian Processes offer a non-parametric approach to regression, providing not only a point estimate but also a measure of uncertainty. This is particularly useful in healthcare applications like CGM data forecasting, where understanding the uncertainty associated with predictions can have clinical significance. The choice of kernel function in a GP model is crucial as it defines the shape and smoothness of the function that the model will learn.

- Support Vector Machines are a class of supervised learning models well-suited for classification and regression tasks. In the context of CGM data, SVMs can provide a robust and efficient mechanism for glucose level prediction. The effectiveness of an SVM model largely hinges on the choice of kernel, which determines the decision boundary. The 'RBF' (Radial Basis Function) kernel is often used for non-linear data, making it a good fit for complex physiological data like CGM readings.

For a great in-depth explanation of GP and SVM we refer the reader to the excellent scikit learn Docs.[^1] and [^2]

While for GP we trained a single model, we trained multiple SVM and took advantage of scikit learn *RegressorChain*. It consists of a multi-label model that arranges regressions into a chain.
Each model makes a prediction in the order specified by the chain using all of the available features provided to the model plus the predictions of models that are earlier in the chain. This can capture some interdependencies between the output variables, if they exist-- in our use case, we expect future BG reading to have non-trivial dependencies from previous ones. We can actually see that this strategy pays off, not only when compared to results from a single SVM, but also when putting our RegressorChain against more complex and nuanced models.

Finally, we should also note that, due to the long convergence time of these algorithms, especially the GP one, these baseline models were trained on a subset of the available data. The selected Hyperparameters for the baseline models are as follows:

### GP Hyperparameters

Hyperparameters for Gaussian Processes

The kernel used in our GP model is a composite function, formulated as:

$$ 
\text{Kernel}=\text{Constant}×\text{RBF}+\text{White Noise} 
$$

| Hyperparameter       | Value      | Bounds          |
|----------------------|------------|-----------------|
| Constant             | 1.0        | [0.0001, 5000]  |
| RBF                  | 10         | [0.001, 1000]   |
| White Noise          | 0.01       | [0.00001, 1]    |

### SVM Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Kernel         | RBF   |
| C              | 1.0   |
| Gamma          | auto  |

Aside from the scores reported [in the overview](https://francesco-vaselli.github.io/GlucoseGuard/overview1/), we show here the Confusion Matrix and a regression example for our baseline models

[^1]: https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process 
[^2]: https://scikit-learn.org/stable/modules/svm.html#svm