In the realm of machine learning for healthcare, class imbalance is a recurrent issue. The minority class—here, critical hypoglycemic events—is often underrepresented, which could lead to poor generalization and biased predictions towards the majority class. Augmentation strategies serve as vital countermeasures, enhancing the robustness of the model by artificially inflating the training dataset with synthetic but plausible examples. This is particularly crucial for applications like CGM data forecasting, where overlooking a rare hypoglycemic event could have serious health implications.

## Augmentation Strategies
### Gaussian Noise

To mitigate overfitting and to make the model more resilient to the innate noise in CGM data, Gaussian noise with a mean of 0 and a variance of 3 was added solely to the input sequences during training. This technique is predicated on the notion that similar input sequences should yield similar behavior; therefore, small perturbations around a true reading should not dramatically alter the forecast. This injects an element of stochasticity into the model, helping it to generalize better to unseen data.

### MixUp

Originating from the work by Zhang et al[^1], MixUp is an augmentation strategy that has garnered attention for its efficacy in enhancing neural network generalization. The method involves linearly interpolating between samples in the training dataset according to the formula:

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y}=\lambda y_i + (1 - \lambda) y_j
$$


Here, $\lambda$ is a hyperparameter following the Beta distribution, Beta($\alpha$,$\alpha$), and $x_i$, $x_j$​ denote inputs from two different samples, respectively. In our application, we constrain MixUp to the minority class, specifically those with $(x, \; y)<80$, to counterbalance the dataset.

The hyperparameter $\alpha$ is a sensitive knob controlling the diversity of synthetic samples. Higher values produce samples more resembling to the reference real data while lower ones introduce samples very different from the reference real data.  The reference work examined $\alpha = 0.4$ and $\alpha = 2$ in twofold MixUp and found improvements in both positive predictive value (PPV) and sensitivity for the minority class across various prediction horizons. We employed $\alpha = 2$.

[^1]: https://arxiv.org/abs/1710.09412