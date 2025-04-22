```python
cd ..
```

    d:\min\research_projects\FedImpute
    

    d:\min\research_projects\FedImpute\.venv\Lib\site-packages\IPython\core\magics\osm.py:417: UserWarning: This is now an optional IPython functionality, setting dhist requires you to install the `pickleshare` library.
      self.shell.db['dhist'] = compress_dhist(dhist)[-100:]
    

# Load Dataset

We first load `codrna` dataset from fedimpute.


```python
%load_ext autoreload
%autoreload 2
from fedimpute.data_prep import load_data, display_data
data, data_config = load_data("codrna")
display_data(data)
print("Data Dimensions: ", data.shape)
print("Data Config:\n", data_config)
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    +--------+--------+--------+--------+--------+--------+--------+--------+--------+
    |   X1   |   X2   |   X3   |   X4   |   X5   |   X6   |   X7   |   X8   |   y    |
    |--------+--------+--------+--------+--------+--------+--------+--------+--------|
    | 0.7554 | 0.1364 | 0.0352 | 0.4132 | 0.6937 | 0.1591 | 0.3329 | 0.7154 | 1.0000 |
    | 0.7334 | 0.7879 | 0.3819 | 0.3693 | 0.5619 | 0.4830 | 0.4351 | 0.5160 | 0.0000 |
    | 0.7752 | 0.1364 | 0.1761 | 0.3290 | 0.7410 | 0.4259 | 0.4644 | 0.5268 | 1.0000 |
    | 0.5905 | 0.7424 | 0.2720 | 0.2898 | 0.6920 | 0.3205 | 0.4019 | 0.6290 | 1.0000 |
    | 0.7366 | 0.1212 | 0.2465 | 0.3290 | 0.7410 | 0.3249 | 0.5086 | 0.5631 | 1.0000 |
    +--------+--------+--------+--------+--------+--------+--------+--------+--------+
    Data Dimensions:  (5000, 9)
    Data Config:
     {'target': 'y', 'task_type': 'classification', 'natural_partition': False}
    

# Construct a distributed data scenario

We then construct a distributed data scenario with 4 clients and heterogenous MNAR missingness.


```python
%load_ext autoreload
%autoreload 2
from fedimpute.scenario import ScenarioBuilder

scenario_builder = ScenarioBuilder()
scenario_data = scenario_builder.create_simulated_scenario(
    data, data_config, num_clients = 4, dp_strategy='iid-even', ms_scenario='mnar-heter'
)
print('Results Structure (Dict Keys):')
print(list(scenario_data.keys()))
scenario_builder.summarize_scenario()
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    Missing data simulation...
    Results Structure (Dict Keys):
    ['clients_train_data', 'clients_test_data', 'clients_train_data_ms', 'clients_seeds', 'global_test_data', 'data_config', 'stats']
    ==================================================================
    Scenario Summary
    ==================================================================
    Total clients: 4
    Global Test Data: (500, 9)
    Missing Mechanism Category: MNAR (Self Masking Logit)
    Clients Data Summary:
         Train     Test      Miss     MS Ratio    MS Feature    Seed
    --  --------  -------  --------  ----------  ------------  ------
    C1  (1125,9)  (113,9)  (1125,8)     0.47         8/8        6077
    C2  (1125,9)  (113,9)  (1125,8)     0.51         8/8        577
    C3  (1125,9)  (113,9)  (1125,8)     0.46         8/8        7231
    C4  (1125,9)  (113,9)  (1125,8)     0.47         8/8        5504
    ==================================================================
    
    

# Build New Imputer

In the following example, we develop a new imputer MICE imputation with a 2 layer neural network as underlying machine learning model for imputation, it should inherit the abstract class `BaseMLImputer` and implement all its abstract methods. It also inherit `ICEImputerMixin` class which contains some helper function for ICE imputation. We add comment in class to give more instructions on how we implement it.


```python
from fedimpute.execution_environment.imputation.base import BaseMLImputer, ICEImputerMixin
from sklearn.neural_network import MLPRegressor
import numpy as np
from collections import OrderedDict

class MLPICEImputer(BaseMLImputer, ICEImputerMixin):
    
    def __init__(self):
        
        super().__init__(name='mlp_mice', model_persistable=False) # it needs two parameters: name of imputer and whether the model is persistable (can be saved to disk), we set it to False because ICE imptutation is not persistable
        
        self.imp_models = [] # list of imputation models (each for a feature)
        self.min_values = None # min values of features used for clipping
        self.max_values = None # max values of features used for clipping
        self.seed = None # seed for randomization
        self.fit_res_history = {} # fit results history

    def initialize(
            self, X: np.array, missing_mask: np.array, data_utils: dict, params: dict, seed: int
    ) -> None:
        """
        Initialize imputer - statistics imputation models etc.

        Args:
            X: data with intial imputed values
            missing_mask: missing mask of data
            data_utils: data utils dictionary - contains information about data
            params: params for initialization
            seed: int - seed for randomization
        """
        
        # initialized imputation models (from sklearn's MLPRegressor (fully connected neural network))
        self.imp_models = []
        for i in range(data_utils['n_features']):
            estimator = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
            X_train = X[:, np.arange(X.shape[1]) != i][0:10]
            y_train = X[:, i][0:10]
            estimator.fit(X_train, y_train)
            self.imp_models.append(estimator)

        # initialize min max values for a clipping threshold (this method is defined in `ICEImputerMixin`)
        self.min_values, self.max_values = self.get_clip_thresholds(data_utils)
        self.seed = seed
    
    def get_imp_model_params(self, params: dict) -> OrderedDict:
        """
        Return model parameters

        Args:
            params: dict contains parameters for get_imp_model_params

        Returns:
            OrderedDict - model parameters dictionary
        """
        # This method is used to get the parameters of the imputation model for a given feature
        # get feature index of imputation models
        feature_idx = params['feature_idx']
        imp_model = self.imp_models[feature_idx]
        
        # get parameters from sklearn model
        coefs = imp_model.coefs_
        intercept = imp_model.intercepts_
        
        # convert parameters to a dictionary (we need to convert parameters to ordered dictionary as required by `BaseMLImputer`)
        parameters = {}
        for i in range(len(coefs)):
            parameters[f'coef_{i}'] = coefs[i]
            parameters[f'intercept_{i}'] = intercept[i]

        return OrderedDict(parameters)

    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:
        """
        Set model parameters

        Args:
            updated_model_dict: global model parameters dictionary
            params: parameters for set parameters function
        """
        # This method is used to set the parameters of the imputation model for a given feature (update models)
        # get feature index of imputation models
        feature_idx = params['feature_idx']
        imp_model = self.imp_models[feature_idx]

        # set parameters to sklearn model
        coefs = []
        intercepts = []
        for i in range(len(imp_model.coefs_)):
            coefs.append(updated_model_dict[f'coef_{i}'])
            intercepts.append(updated_model_dict[f'intercept_{i}'])
        imp_model.coefs_ = coefs
        imp_model.intercepts_ = intercepts

    def fit(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> dict:
        """
        Fit imputer to train local imputation models

        Args:
            X: np.array - float numpy array features
            y: np.array - target
            missing_mask: np.array - missing mask
            params: parameters for local training
        """
        # This method is used to fit the imputation model for a given feature
        # get complete data of the feature
        feature_idx = params['feature_idx']
        row_mask = missing_mask[:, feature_idx]
        X_train = X[~row_mask][:, np.arange(X.shape[1]) != feature_idx]
        y_train = X[~row_mask][:, feature_idx]

        # fit MLP imputation models
        estimator = self.imp_models[feature_idx]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_train)
        loss = np.mean((y_pred - y_train) ** 2)
        self.fit_res_history[feature_idx].append({
            'loss': loss,
            'sample_size': X_train.shape[0]
        })

        return {
            'loss': loss,
            'sample_size': X_train.shape[0]
        }

    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:
        """
        Impute missing values using an imputation model

        Args:
            X (np.array): numpy array of features
            y (np.array): numpy array of target
            missing_mask (np.array): missing mask
            params (dict): parameters for imputation

        Returns:
            np.ndarray: imputed data - numpy array - same dimension as X
        """
        # This method is used to impute the missing values using the imputation model for a given feature
        # get incomplete data of the feature
        feature_idx = params['feature_idx']
        row_mask = missing_mask[:, feature_idx]
        if np.sum(row_mask) == 0:
            return X

        # impute missing values
        X_test = X[row_mask][:, np.arange(X.shape[1]) != feature_idx]
        estimator = self.imp_models[feature_idx]
        imputed_values = estimator.predict(X_test)
        imputed_values = np.clip(imputed_values, self.min_values[feature_idx], self.max_values[feature_idx])
        X[row_mask, feature_idx] = np.squeeze(imputed_values)

        return X
    
    def get_fit_res(self, params: dict) -> dict:

        # This method is used to get fit results for a given feature from the fit history
        try:
            feature_idx = params['feature_idx']
        except KeyError:
            raise ValueError("Feature index not found in params")
        
        return self.fit_res_history[feature_idx][-1]
```

# Register New Imputer to Environment

Once finishing develop new imputer, it needs to be registered into fedimpute, so it can be used by constructed environment. We need to call `register_imputer` method from `env.register` object. It takes name of imputer, class of imputer, workflow associated with imputer and a list of supported federated strategy of imputer.


```python
%load_ext autoreload
%autoreload 2
from fedimpute.execution_environment import FedImputeEnv

env = FedImputeEnv(debug_mode=False)
env.register.register_imputer(
	name = 'mlp_mice',                     # name of we give to the new imputer
	imputer = MLPICEImputer,              # the class of the new imputer we just developed
	workflow = 'ice',                     # because it is ICE imputation, we use 'ice' workflow
	fed_strategy = ['local', 'fedmice']   # we support local and fedmice strategy for this imputer
)
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    


```python
# then we can use the new imputer in the environment and run the federated imputation
env.configuration(imputer = 'mlp_mice', fed_strategy='local', workflow_params={'log_metric': None})
env.setup_from_scenario_builder(scenario_builder = scenario_builder, verbose=1)
env.show_env_info()
env.run_fed_imputation(verbose=2)
```

    [1mSetting up clients...[0m
    [1mSetting up server...[0m
    [1mSetting up workflow...[0m
    [1mEnvironment setup complete.[0m
    ============================================================
    Environment Information:
    ============================================================
    Workflow: ICE (Imputation via Chain Equation)
    Clients:
     - Client 0: imputer: mlp_mice, fed-strategy: local
     - Client 1: imputer: mlp_mice, fed-strategy: local
     - Client 2: imputer: mlp_mice, fed-strategy: local
     - Client 3: imputer: mlp_mice, fed-strategy: local
    Server: fed-strategy: local
    ============================================================
    
    [32m[1mImputation Start ...[0m
    [1mInitial: imp_rmse: 0.1658 imp_ws: 0.0827 [0m
    


    ICE Iterations:   0%|          | 0/20 [00:00<?, ?it/s]



    Feature_idx:   0%|          | 0/8 [00:00<?, ?it/s]


    [1mEpoch 0: imp_rmse: 0.2145 imp_ws: 0.1100 loss: 0.0074 [0m
    


    Feature_idx:   0%|          | 0/8 [00:00<?, ?it/s]


    [1mEpoch 1: imp_rmse: 0.2279 imp_ws: 0.1138 loss: 0.0065 [0m
    


    Feature_idx:   0%|          | 0/8 [00:00<?, ?it/s]


    [1mEpoch 2: imp_rmse: 0.2276 imp_ws: 0.1131 loss: 0.0059 [0m
    


    Feature_idx:   0%|          | 0/8 [00:00<?, ?it/s]


    [1mEpoch 3: imp_rmse: 0.2351 imp_ws: 0.1153 loss: 0.0061 [0m
    


    Feature_idx:   0%|          | 0/8 [00:00<?, ?it/s]


    [1mEpoch 4: imp_rmse: 0.2383 imp_ws: 0.1164 loss: 0.0062 [0m
    


    Feature_idx:   0%|          | 0/8 [00:00<?, ?it/s]


    [1mEpoch 5: imp_rmse: 0.2378 imp_ws: 0.1158 loss: 0.0059 [0m
    


    Feature_idx:   0%|          | 0/8 [00:00<?, ?it/s]


    [1mEpoch 6: imp_rmse: 0.2397 imp_ws: 0.1158 loss: 0.0059 [0m
    


    Feature_idx:   0%|          | 0/8 [00:00<?, ?it/s]


    [1mEpoch 7: imp_rmse: 0.2432 imp_ws: 0.1186 loss: 0.0063 [0m
    


    Feature_idx:   0%|          | 0/8 [00:00<?, ?it/s]


    [1mEpoch 8: imp_rmse: 0.2389 imp_ws: 0.1167 loss: 0.0063 [0m
    [1mAll clients converged, iteration 8[0m
    [1mFinal: imp_rmse: 0.2389 imp_ws: 0.1167 [0m
    [32m[1mFinished. Running time: 65.1471 seconds[0m
    
