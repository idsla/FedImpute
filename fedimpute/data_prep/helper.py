from typing import Tuple, Union, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate


def one_hot_encoding(
        data: Union[pd.DataFrame, np.ndarray], numerical_cols_num: int, max_cateogories: int = 10
) -> np.ndarray:

    one_hot_encoder = OneHotEncoder(
        categories='auto', handle_unknown='ignore', drop='first', max_categories=max_cateogories
    )

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    categorical_cols = [
        data.columns[col_idx] for col_idx in range(data.shape[1]) if numerical_cols_num < col_idx < data.shape[1] - 1
    ]

    one_hot_encoder.fit(data[categorical_cols])
    one_hot_encoded = one_hot_encoder.transform(data[categorical_cols]).toarray()
    one_hot_encoded = pd.DataFrame(one_hot_encoded)
    data_num = data.drop(categorical_cols + [data.columns[-1]], axis=1).reset_index(drop=True)
    data_target = data[data.columns[-1]].reset_index(drop=True)
    data = pd.concat([data_num, one_hot_encoded, data_target], axis=1)

    return data.values


def ordering_features(
        data: Union[pd.DataFrame, np.ndarray], numerical_cols: Union[List[int], List[str]], target_col: Union[str, int]
) -> pd.DataFrame:
    """
    Ordering the features in the data - numerical columns first, then categorical columns, and finally the target column.
    :param data: data to be ordered - can be a pandas dataframe or a numpy array
    :param numerical_cols: list of numerical columns name (pandas) or their indices (numpy)
    :param target_col: target column name (pandas) or its index (numpy)
    :return:
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
        if isinstance(numerical_cols[0], str):
            raise ValueError("numerical_cols should be a list of integers when data is a numpy array")
        if target_col == -1:
            target_col = data.shape[1] - 1

    elif isinstance(data, pd.DataFrame):
        if isinstance(numerical_cols[0], int):
            numerical_cols = [data.columns[i] for i in numerical_cols]
        if isinstance(target_col, int):
            target_col = data.columns[target_col]

    else:
        raise ValueError("data should be a pandas dataframe or a numpy array")

    categorical_cols = [col for col in data.columns if col not in numerical_cols]
    data = data[numerical_cols + categorical_cols + [target_col]]

    return data


def display_data(data: pd.DataFrame):
    print(
        tabulate(
            data.head(), 
            headers='keys', 
            showindex=False, 
            tablefmt="psql", 
            numalign="center", 
            stralign="center",
            floatfmt=".4f"
        )
    )

