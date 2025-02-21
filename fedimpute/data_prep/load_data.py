import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler
)

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
            #df['fbs']  = df['fbs'].fillna(df['fbs'].mean())
            #df['trestbps'] = df['trestbps'].fillna(df['trestbps'].mean())
            #df['chol'] = df['chol'].fillna(df['chol'].mean())
            #df['thalach'] = df['thalach'].fillna(df['thalach'].mean())
            #df['exang'] = df['exang'].fillna(df['exang'].mean())
            #df = df.drop(columns=['ca', 'slope', 'thal'])
            #df = df.dropna()
            #df['fbs'] = df['fbs'].astype(int)
            #df['sex'] = df['sex'].astype(int)
            #df['cp'] = df['cp'].astype(int)
            #df['restecg'] = df['restecg'].astype(int)
            #print(site, df.shape)
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

        #cat_cols = ['sex', 'cp', 'restecg', 'fbs']
        # target_col = 'num'
        # num_cols = [col for col in dfs[0].columns if col != target_col and col not in cat_cols]

        # df = pd.concat(dfs, axis=0)
        # client_split_indices = np.cumsum([df_sub.shape[0] for df_sub in dfs[:-1]])
        # print(client_split_indices)

        # # standardize
        # from sklearn.preprocessing import StandardScaler, MinMaxScaler

        # scaler = StandardScaler()
        # df[num_cols] = scaler.fit_transform(df[num_cols])
        # scaler = MinMaxScaler()
        # df[num_cols] = scaler.fit_transform(df[num_cols])

        # df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # print(df.shape)
        # print(num_cols)
        # cat_cols = [col for col in df.columns if col not in num_cols + [target_col]]
        # df = df[num_cols + cat_cols + [target_col]]
        # for col in cat_cols:
        #     df[col] = df[col].astype(int)
            
        # df[target_col] = df[target_col].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})

        # data_config = {
        #     'target': target,
        #     'important_features_idx': [
        #         idx for idx in range(0, df.shape[1]) 
        #         if (df.columns[idx] != target) and (df.columns[idx] not in cat_cols)
        #     ],
        #     'features_idx': [idx for idx in range(0, df.shape[1]) if df.columns[idx] != target],
        #     "num_cols": df.shape[1] - 1 - len(cat_cols),
        #     'task_type': 'classification',
        #     'clf_type': 'binary-class',
        #     'data_type': 'tabular',
        #     'client_split_indices': client_split_indices.tolist()
        # }
        # print(data_config)

        # with open('./data/heart_disease/data_config_binary.json', 'w') as f:
        #     json.dump(data_config, f)

        # df.to_csv('./data/heart_disease/data_clean_binary.csv', index=False)
    else:
        raise ValueError(f"Data {data_name} not found")
    
    return data


