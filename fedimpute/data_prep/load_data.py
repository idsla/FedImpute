import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler
)
from sklearn.pipeline import Pipeline

from fedimpute.data_prep.helper import download_data

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
    
    elif data_name == 'fed_heart_disease':
        
        download_data('https://archive.ics.uci.edu/static/public/45/heart+disease.zip', 'heart_disease')
        
        # load federated data
        dfs = []
        for site in ['cleveland', 'hungarian', 'switzerland', 'va']:
            df = pd.read_csv('./data/heart_disease/processed.{}.data'.format(site), header=None, na_values='?')
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
            df.columns = columns
            dfs.append(df)
        
        df = pd.concat(dfs, axis=0).reset_index(drop=True)
        split_indices = np.cumsum([0] + [df_sub.shape[0] for df_sub in dfs])
        
        cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'thal']
        num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']
        drop_cols = ['ca']
        target_col = 'num'
        
        df = df.drop(columns=drop_cols)
        df_features = df[num_cols + cat_cols].copy()
        for col in cat_cols:
            df_features[col] = df_features[col].fillna(-1)
        
        df_features = pd.get_dummies(df_features, columns=cat_cols, drop_first=True)
        df_target = df[target_col].copy()
        df_target = df_target.apply(lambda x: 0 if x == 0 else 1)
        
        cat_cols = [col for col in df_features.columns if col not in num_cols]
        
        scaler = StandardScaler()
        df_features[num_cols] = scaler.fit_transform(df_features[num_cols])
        scaler = MinMaxScaler()
        df_features[num_cols] = scaler.fit_transform(df_features[num_cols])
        
        data = pd.concat([df_features, df_target], axis=1)
        
        data_config = {
            'target': target_col,
            'task_type': 'classification',            
            'natural_partition': True,
        }
        
        dfs = [data.iloc[
            split_indices[i]:split_indices[i+1]].reset_index(drop=True).copy() for i in range(len(split_indices)-1)
        ]
          
        return dfs, data_config

    if data_name == "california":
        
        housing = fetch_california_housing()
        data = pd.DataFrame(data=housing.data, columns=housing.feature_names)
        target_col = 'MedHouseVal'
        data[target_col] = housing.target

        # drop missing values
        data = data.dropna()

        # remove outliers
        data = outlier_remove_iqr(data, 'AveRooms')
        data = outlier_remove_iqr(data, 'AveBedrms')
        data = outlier_remove_iqr(data, 'Population')
        data = outlier_remove_iqr(data, 'AveOccup')

        # gaussian transform
        data = convert_gaussian(data, 'MedInc')

        num_cols = data.columns.tolist()[:-1]

        scaler = Pipeline([
            ('standard', StandardScaler()),
            ('minmax', MinMaxScaler())
        ])

        data[num_cols] = scaler.fit_transform(data[num_cols])
        
        data_config = {
            'target': target_col,
            'task_type': 'regression',            
            'natural_partition': False,
        }
        
        data = data.sample(n=5000, random_state=42).reset_index(drop=True)
        
        return data, data_config
    else:
        raise ValueError(f"Data {data_name} not found")
    
    return data


