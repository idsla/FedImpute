import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_data(data_name: str):
    
    if data_name == "codrna":
        features, labels = fetch_openml(data_id = 351, as_frame='auto', return_X_y = True)
        df_pred = pd.DataFrame.sparse.from_spmatrix(features).sparse.to_dense()
        df_pred.columns = [f"X{i+1}" for i in range(df_pred.shape[1])]
        df_label = pd.DataFrame(labels)
        df_label = pd.factorize(df_label[0])[0]
        df_label = pd.DataFrame(df_label, columns=["y"]).astype(int)
        data = pd.concat([df_pred, df_label], axis=1)
        data_standard = StandardScaler().fit_transform(data.values)
        data_minmax = MinMaxScaler().fit_transform(data_standard)
        data = pd.DataFrame(data_minmax, columns=data.columns)
        data_config = {
            'target': 'y',
            'task_type': 'classification',            
            'natural_partition': False,
        }
        
        data = data.sample(n=5000, random_state=42).reset_index(drop=True)
        
        return data, data_config
    
    elif data_name == "california":
        pass
    elif data_name == 'hhnp':
        pass
    elif data_name == 'dvisits':
        pass
    elif data_name == 'condon':
        pass
    elif data_name == 'school':
        pass
    elif data_name == 'vehicle':
        pass
    else:
        raise ValueError(f"Data {data_name} not found")
    
    return data


