# Dataset and Preprocessing

The first step for using FedImpute is to prepare the data. 

## Input Data Format and Preprocessing

The data should be tabular data in the form of a **numpy array** (`<np.ndarray>`) or **List of numpy arrays** for those ***naturally partitioned federated data***, where each row represents an observation and each column represents a feature. 
It will be the input to the simulation process, where it will be partitioned into subset as local dataset for each party and the missing data will be introduced. Currently, FedImpute only supports the numerical typed data, for categorical data, you need to one-hot encode them into binary features.

### Required Preprocessing Steps

There are some basic **preprocessing steps** that you need to follow before using FedImpute, 
The final dataset should be in the form of a numpy array with the columns ordered as follows format: 

```text
| --------------------- | ------------------ | ------ |
| numerical features... | binary features... | target |
| --------------------- | ------------------ | ------ |
| 0.1 3 5 ...           | 1 0 1 0 0 0        | ...    |
...
| 0.5 10 1 ...          | 0 0 1 0 0 1        | ...    |
| --------------------- | ------------------ | ------ |
```

###### Ordering Features

To facilitate the ease of use for FedImpute, you have to order the features in the dataset such that the **numerical features** are placed first, followed by the **binary features**.
The **target variable** should be the ***last*** column in the dataset.

###### One-hot Encoding Categorical Features 

Currently, FedImpute only supports **numerical** and **binary** features, does ***not*** support **categorical** features in the dataset.
So you have to one-hot encode the categorical features into binary features before using FedImpute.

###### Data Normalization (Optional)

It is recommended to normalize the numerical features in the dataset within range of 0 and 1.

### Helper Functions for Preprocessing

FedImpute provides several helper functions to perform the required preprocessing steps. Example of the helper functions are as follows:

```python
from fedimpute.data_prep.helper import ordering_features, one_hot_encoding

# Example for data with numpy array 
data = ...
data = ordering_features(data, numerical_cols=[0, 1, 3, 4, 8], target_col=-1)
data = one_hot_encoding(data, numerical_cols_num=5, max_cateogories=10)

# Example data with pandas dataframe
data = ...
data = ordering_features(
    data, numerical_cols=['age', 'income', 'height', 'weight', 'temperature'], 
    target_col='house_price'
)
data = one_hot_encoding(data, numerical_cols_num=5, max_cateogories=10)

```
- `ordering_features(data, numerical_cols: List[str or int], target_col: int or str)`: This function will order the features in the dataset such that the numerical features are placed first, followed by the binary features. The target variable should be the last column in the dataset.
- `one_hot_encoding(data, numerical_cols_num: int)`: This function will one-hot encode the categorical features into binary features. It assumes you data is already orderd as numerical cols + cat_cols + target, so You just need to specify the number of numerical columns.
***Note***: The `ordering_features` function is required to be called before the `one_hot_encoding` function.

We also provide a one-for-all function to perform all the preprocessing steps at once. 
```python
from fedimpute.data_prep import prep_data

data = ...
data = prep_data(
    data, numerical_cols=['age', 'income', 'height', 'weight', 'temperature'], target_col='house_price'
)

```

## Data Configuration Dictionary
To allow FedImpute to understand the data and the task type, you need to provide a configuration dictionary called `data_config`.
The example of the `data_config` dictionary is as follows:

```python
data_config = {
    'target': 'house_price',
    'task_type': 'classification',
    'natural_partition': False
}
```

The `data_config` dictionary should contain the following keys:

- `target`: The target variable name. 
- `task_type`: The task type of the target variable. It can be either `classification` or `regression`.
- `natural_partition`: Whether the data is naturally partitioned into different parties. If it is, set it to `True`. Otherwise, set it to `False`.
