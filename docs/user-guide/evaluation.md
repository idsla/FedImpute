
# Evaluation of Imputation Outcomes

Fedimpute provides a comprehensive evaluation module to assess the effectiveness of federated imputation algorithms across various missing data scenarios. 
The evaluation can be categorized into the following aspects:

- **Imputation Quality**: Evaluate the quality of imputed data.
- **Downstream Prediction**: Evaluate the performance of downstream prediction tasks using imputed data (supports both local or federated prediction).

## Basic Usage

The `Evaluator` class is the `evaluation` module's main class, use its `evaluation()` function to perform evaluation.

```python
from fedimpute.evaluation import Evaluator

evaluator = Evaluator()
evaluator.evaluate(env, ['imp_quality', 'pred_downstream_local', 'pred_downstream_fed'])
evaluator.show_results()
```

The `Evaluator.evaluate()` method is used to evaluate the imputation outcomes. It takes the `FedImpEnv` object (see [Federated Imputaton](../user-guide/fed_imp.md) and a list of evaluation aspects as input.
The evaluation aspects can be one or more of the following:

- `imp_quality`: Evaluate the quality of imputed data.
- `pred_downstream_local`: Evaluate the performance of downstream prediction tasks using imputed data in a local setting.
- `pred_downstream_fed`: Evaluate the performance of downstream prediction tasks using imputed data in a federated setting.

The `Evaluator.show_results()` method is used to display the evaluation results. It prints the evaluation results for each evaluation aspect.


## Supported Evaluation Metrics

The following evaluation metrics are supported for each evaluation aspect:

### Imputation Quality

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

### Downstream Prediction

After missing data are imputed, 
the downstream task prediction can be performed on imputed data. 
During the data partition stage, we retain a local test dataset for each client and a global test dataset for global data. 
These test datasets can be used to evaluate downstream prediction models trained on clients' local imputed datasets to measure 
the goodness of imputation and how it influences the prediction.  
We contains three built-in local downstream prediction models (Linear regression `linear`, Random Forest `tree`, and two-layer neural network `nn`) 
and various metrics such as accuracy `accu`, AUROC `auroc`, f1 score `f1` for evaluation of classification tasks, and 
mean square error `mse`, r2 score `r2`, mean square log error `msle`, mean abusolute error `mae` for 
evaluation of regression tasks. Fedimpute also supports federated downstream task evaluation for a 
two-layer neural network with FedAvg `fedavg` as federated learning strategies. By assessing the performance of the downstream tasks, we can gain insights into how different imputation methods impact the overall effectiveness of the learning process in the presence of missing data.

### Variance Metrics (Measure the variance of imputationa and prediction outcomes)

- **Variance `variance`**: Variance is a statistical measure that quantifies the spread of 
a set of values around their mean. We calculate the variance of each 
client's imputation quality metric value. 

- **Jain Index `jain-index`**:The Jain index, 
also known as the Jain fairness index, is a metric used to evaluate the fairness or 
equality of resource allocation in distributed systems. We adopt this metric to assess the equality of imputation 
quality across clients. The Jain index ranges from $ 1/N $
to $1$, where $1$ is the number of clients. A value of $1$ indicates perfect equality, 
meaning that all clients have the same imputation quality, while a value of $1/N$ 
- represents the worst-case scenario, where one client dominates the imputation performance. 

- **Entropy `entropy` **: We utilize entropy 
to capture the distributional variation of imputation quality across clients.
We compute the entropy of this distribution using the standard formula: $-\sum(p_i * log(p_i))$, where $p_i$ is the normalized imputation 
quality metric value for client $i$. A higher entropy value indicates a more 

