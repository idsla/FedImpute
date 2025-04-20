import numpy as np
import pandas as pd
from typing import List

def dataframe_to_numpy(data: pd.DataFrame, config: dict) -> np.array:
    """
    Convert a pandas DataFrame to a numpy array with the target column at the last column.
    """
    target_col = config['target']
    columns = data.columns.tolist()
    columns.remove(target_col)
    columns.append(target_col)
    
    for col in columns:
        if data[col].dtype == 'object':
            print('Warning: Object type column detected. Input data should not contain object or categorical type columns.')
            data[col] = pd.factorize(data[col])[0].astype(int)
    
    columns = data.columns.tolist()
    data = data[columns].to_numpy()
    
    return data, columns

def arrays_to_dataframes(data: List[np.array], columns: List[str], without_target: bool = False) -> List[pd.DataFrame]:
    if without_target:
        columns = columns[:-1]
    return [pd.DataFrame(data[i].copy(), columns=columns).astype(float) for i in range(len(data))]

