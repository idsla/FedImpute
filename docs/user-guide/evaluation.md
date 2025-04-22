
# Evaluation of Imputation Outcomes

Fedimpute provides a comprehensive evaluation module to assess the effectiveness of federated imputation algorithms across various missing data scenarios. 
The evaluation can be categorized into the following aspects:

- **Imputation Quality**: Evaluate the quality of imputed data.
- **Local Prediction**: Evaluate the performance based on downstream local prediction tasks using imputed data.
- **Federated Prediction**: Evaluate the performance based on downstream federated prediction task using imputed data.

## Basic Usage

The `Evaluator` class is the `evaluation` module's main class, use its `evaluation()` function to perform evaluation.

```python
from fedimpute.evaluation import Evaluator

evaluator = Evaluator()
ret = evaluator.evaluate_all(
    env, metrics = ['imp_quality', 'pred_downstream_local', 'pred_downstream_fed']
)
evaluator.show_results_all()
```

The `Evaluator.evaluate_all()` method is used to evaluate the imputation outcomes. It takes the `FedImpEnv` object (see [Federated Imputaton](../user-guide/fed_imp.md) and a list of evaluation aspects as input.
The evaluation aspects can be one or more of the following:

- `imp_quality`: Evaluate the quality of imputed data.
- `pred_downstream_local`: Evaluate the performance of downstream prediction tasks using imputed data in a local setting.
- `pred_downstream_fed`: Evaluate the performance of downstream prediction tasks using imputed data in a federated setting.

The `Evaluator.show_results_all()` method is used to display the evaluation results. It prints the evaluation results for each evaluation aspect.


## Supported Evaluation

The following evaluation metrics are supported for each evaluation aspect:

### Imputation Quality

User can use the specific `evaluate_imputation_quality()` method in `evaluation.Evaluator` class provides functionalities to evaluate the quality of imputed data across clients comprehensively. It has several parameters: 

- `X_train_imps`: lists of client-specific imputed datasets
- `X_train_origins`: list of client-specific original complete datasets
- `X_train_masks`: list of client-specific missing value masks
- `metrics`: to denote the list of metrics for evaluation.

**Metrics:**

- **Root Mean Squared Error (RMSE) `rmse`**: RMSE is calculated by taking the square root of the mean of 
- the squared differences between the imputed and original values. A lower RMSE indicates better imputation accuracy.

- **Normalized RMSE `nrmse`**: Normalized RMSE is an extension of the standard RMSE that 
allows for a more intuitive interpretation and comparison of imputation qualities. It is 
calculated by dividing the RMSE by the range (i.e., standard deviation) of the original data. 
This normalization process scales the RMSE to a value between 0 and 1 to provide a standardized metric independent of the data scale.

- **Sliced Wasserstein Distance `sliced-ws`**: Sliced Wasserstein distance is a metric 
that measures the dissimilarity between two high-dimensional probability distributions. 
We use sliced Wasserstein distance to assess the discrepancy between the probability distributions
of the imputed data and the original data for each client.
A smaller Wasserstein distance indicates a higher similarity between the imputed and original data distributions. 

User can use `show_imp_results()` to get the formatted results of evaluation.

```{python}
from fedimpute.evaluation import Evaluator

X_trains = env.get_data(client_ids='all', data_type = 'train')
X_train_imps = env.get_data(client_ids='all', data_type = 'train_imp')
X_train_masks = env.get_data(client_ids='all', data_type = 'train_mask')

evaluator = Evaluator()
ret = evaluator.evaluate_imp_quality(
    X_train_imps = X_train_imps,
    X_train_origins = X_trains,
    X_train_masks = X_train_masks,
    metrics = ['rmse', 'nrmse', 'sliced-ws']
)
evaluator.show_imp_results()
```

### Imputation Quality via tSNE visualization

`Evaluator` class also provides a method called `tsne_visualization()` to give the visualized comparison of similarity between the imputed data and the original data (ground-truth data). It visualizes the t-Distributed Stochastic Neighbor Embedding (t-SNE) of imputed data and original data so that the user can visually assess the effectiveness of the imputation outcome. tsne_visualization() takes parameters including client’s imputation data (`X_imp`) and original data (ground-truth data) (`X_origin`) and a random seed (`seed`) used for calculating t-SNE embedding.

```{python}
X_trains = env.get_data(client_ids='all', data_type = 'train')
X_train_imps = env.get_data(client_ids='all', data_type = 'train_imp')

evaluator.tsne_visualization(
    X_imps = X_train_imps,
    X_origins = X_trains,
    seed = 0
)
```

### Local Regression Analysis

The `run_local_regression_analysis()` method in the evaluation.Evaluator class provides functionality for evaluation via local regression analysis tasks. It accepts several parameters: 

- `X_train_imps, y_trains`: lists of client-specific imputed training datasets and targets
- `data_config`: the data configuration dictionary.

The method returns a Dict containing evaluation results. Users can utilize the `show_local_regression_results()` method
in the evaluation.Evaluator class to print a formatted output of the evaluation results.

```{python} 
X_trains, y_trains = env.get_data(client_ids='all', data_type = 'train', include_y=True)
X_train_imps = env.get_data(client_ids='all', data_type = 'train_imp')
data_config = env.get_data(data_type = 'config')

ret = evaluator.run_local_regression_analysis(
    X_train_imps = X_train_imps,
    y_trains = y_trains,
    data_config = data_config
)
evaluator.show_local_regression_results()
```

### Local Prediction

After missing data are imputed,  the downstream task prediction can be performed on imputed data.  During the data partition stage, we retain a local test dataset for each client and a global test dataset for global data.  These test datasets can be used to evaluate downstream prediction models trained on clients' local imputed datasets to measure  the goodness of imputation and how it influences the prediction.  

The `run_local_prediction()` method in the evaluation.Evaluator class provides functionality for evaluation via local prediction tasks. It accepts several parameters: 

- `X_train_imps, y_train`: lists of client-specific imputed training datasets and targets
-  `X_tests, y_tests`: lists of client-specific local test datasets and targets
- `model`: a model specification parameter. The method currently implements three built-in downstream prediction models: linear models
('lr'), random forests ('rf'), and two-layer neural networks ('nn'). 

The method trains prediction models for each client using the imputed training data and evaluates performance on the corresponding test data. For classification tasks, the evaluation metrics include accuracy, F1-score, Area Under the Receiver Operating Characteristic Curve (AUROC), and Area Under
the Precision-Recall Curve (AUPRC). Mean squared error and R2 score are computed for regression tasks.

`show_local_prediction_results()` will give a formatted result summary for the evaluation.

```{python}
X_trains, y_trains = env.get_data(client_ids='all', data_type = 'train', include_y=True)
X_tests, y_tests = env.get_data(client_ids='all', data_type = 'test', include_y=True)
X_train_imps = env.get_data(client_ids='all', data_type = 'train_imp')
data_config = env.get_data(data_type = 'config')

ret = evaluator.evaluate_local_pred(
    X_train_imps = X_train_imps,
    X_train_origins = X_trains,
    y_trains = y_trains,
    X_tests = X_tests,
    y_tests = y_tests,
    data_config = data_config,
    model = 'nn',
    seed= 0
)
evaluator.show_local_prediction_results()
```

### Federated Regression Analysis

The `run_fed_regression_analysis()` method in the evaluation.Evaluator class provides functionality for evaluation via federated regression analysis tasks. It accepts several parameters: 

- `X_train_imps, y_trains`: lists of client-specific imputed training data and targets
- `data_config`: the data configuration dictionary.

The method returns a Dict containing evaluation results. Users can utilize the `show_fed_regression_results()` method
in the evaluation.Evaluator class to print a formatted output of the evaluation results.

```{python}
X_train_imps = env.get_data(client_ids='all', data_type = 'train_imp')
X_trains, y_trains = env.get_data(
    client_ids='all', data_type = 'train', include_y=True
)
data_config = env.get_data(data_type = 'config')

ret = evaluator.run_fed_regression_analysis(
    X_train_imps = X_train_imps,
    y_trains = y_trains,
    data_config = data_config
)
evaluator.show_fed_regression_results()
```

### Federated Prediction

We implement federated prediction functionality by `run_fed_prediction()` method. The current implementation supports federated prediction using a two-layer neural network with Federated Averaging (FedAvg) as the federated learning strategy. We will include more federated models in the future. Similarly, it uses accuracy, F1-Score, AUROC, AUPRC for classification tasks, and mean square error, R2 score for regression tasks. 

It accepts multiple parameters: 

- `X_train_imps, y_trains`: lists of client-specific imputed training data and targets
- `X_tests, y_tests`: lists of client-specific local test data and targets
- `X_test_global, y_test_global`: global test data 
- `model_name`: the name of the model to be used for federated prediction. Currently, federated models including`lr`, `svm`, `rf`, `xgboost`, `nn` are supported.
- `train_params`: the parameters for the federated learning training.
- `model_params`: the parameters for the model.
- `seed`: the random seed for the evaluation.

The method returns a Dict containing evaluation results. Users can utilize the `show_fed_pred_result()` method
in the evaluation.Evaluator class to print a formatted output of the evaluation results.

```{python}
X_train_imps = env.get_data(client_ids='all', data_type = 'train_imp')
X_trains, y_trains = env.get_data(
    client_ids='all', data_type = 'train', include_y=True
)
X_tests, y_tests = env.get_data(
    client_ids='all', data_type = 'test', include_y=True
)
X_global_test, y_global_test = env.get_data(
    data_type = 'global_test', include_y = True
)
data_config = env.get_data(data_type = 'config')

ret = evaluator.run_fed_prediction(
    X_train_imps = X_train_imps,
    X_train_origins = X_trains,
    y_trains = y_trains,
    X_tests = X_tests,
    y_tests = y_tests,
    X_test_global = X_global_test,
    y_test_global = y_global_test,
    data_config = data_config,
    train_params = {
        'global_epoch': 100,
        'local_epoch': 10,
        'fine_tune_epoch': 200,
    },
    seed= 0
)

evaluator.show_fed_prediction_results()
```

## Save Evaluation Results

The evaluation module provides convenient interfaces for presenting and exporting the results. All evaluation functions return results in a dictionary format, which can be formatted into readable tables through dedicated display functions, including `show_imp_results()`,
`show_local_prediction_results()`, `show_fed_prediction_results()` for each evaluation aspects. For further analysis and reporting, the `export_results()` method supports exporting results to different formats, including `pandas.DataFrame` and structured dictionaries.