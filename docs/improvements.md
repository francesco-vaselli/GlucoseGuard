## Possible Improvements to Current Work

### More Inputs: *Insulin on Board*, Activity Reports, Carbohydrates and Food...
- **Why It's Relevant**: Adding additional inputs like insulin on board, activity levels, and dietary intake can provide a more holistic view of the factors affecting glucose levels. This enriched data set could enhance the model's predictive accuracy by accounting for variables that are intrinsically linked to glucose fluctuations.

### Time GAN Data Augmentation
- **Why It's Relevant**: Time-series Generative Adversarial Networks (Time GANs) can simulate realistic, yet synthetic, time-series data. This could be especially valuable for handling imbalances in the dataset, such as under-represented hypoglycemic or hyperglycemic events, thereby improving the model's generalizability.

### Longer Input Sequences
- **Why It's Relevant**: Extending the length of input sequences may capture more nuanced trends and periodicities in glucose levels. This could lead to predictions that are not only more accurate but also more sensitive to long-term physiological changes in an individual patient.

### Longer Prediction Horizons
- **Why It's Relevant**: Lengthening the prediction horizon would allow for more advanced planning and intervention. For example, it could give healthcare providers and patients a better chance to preemptively address hyperglycemic or hypoglycemic episodes, thus improving patient outcomes.

### Tree Class Classification Problem
- **Why It's Relevant**: A three-class classification problem (e.g., hypoglycemic, normoglycemic, hyperglycemic) would offer a more nuanced understanding than a binary above-or-below threshold model. It would enable more targeted interventions based on the severity of the glucose level deviations, making the prediction more actionable.
