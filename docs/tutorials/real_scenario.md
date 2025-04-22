
```python
import numpy as np
import pandas as pd
import tabulate
```

# Load Data


```python
%load_ext autoreload
%autoreload 2

from fedimpute.data_prep import load_data, display_data, column_check
from fedimpute.scenario import ScenarioBuilder
data, data_config = load_data("fed_heart_disease")
scenario_builder = ScenarioBuilder()
scenario_data = scenario_builder.create_real_scenario(
    data, data_config,
)
scenario_builder.summarize_scenario()
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    ==================================================================
    Scenario Summary
    ==================================================================
    Total clients: 4
    Global Test Data: (94, 15)
    Missing Mechanism Category: MCAR
    Clients Data Summary:
         Train     Test      Miss     MS Ratio    MS Feature    Seed
    --  --------  -------  --------  ----------  ------------  ------
    C1  (244,15)  (28,15)  (244,14)     0.00         0/14       6077
    C2  (237,15)  (27,15)  (237,14)     0.05         4/14       577
    C3  (99,15)   (11,15)  (99,14)      0.01         3/14       7231
    C4  (162,15)  (18,15)  (162,14)     0.10         5/14       5504
    ==================================================================
    
    


```python
scenario_builder.visualize_missing_pattern(
    client_ids=[0, 1, 2, 3], data_type='train', fontsize=20, save_path='./plots/real_pattern_train.png'
)
```


```python
scenario_builder.visualize_missing_pattern(
    client_ids=[0, 1, 2, 3], data_type='test', fontsize=20, save_path='./plots/real_pattern_test.png'
)
```

# Running Federated Imputation


```python
%load_ext autoreload
%autoreload 2
from fedimpute.execution_environment import FedImputeEnv

env = FedImputeEnv(debug_mode=False)
env.configuration(imputer = 'mice', fed_strategy='fedmice', workflow_params = {'early_stopping_metric': 'loss'})
env.setup_from_scenario_builder(scenario_builder = scenario_builder, verbose=1)
env.show_env_info()
env.run_fed_imputation(verbose=1)
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    [1mSetting up clients...[0m
    [1mSetting up server...[0m
    [1mSetting up workflow...[0m
    [1mEnvironment setup complete.[0m
    ============================================================
    Environment Information:
    ============================================================
    Workflow: ICE (Imputation via Chain Equation)
    Clients:
     - Client 0: imputer: mice, fed-strategy: fedmice
     - Client 1: imputer: mice, fed-strategy: fedmice
     - Client 2: imputer: mice, fed-strategy: fedmice
     - Client 3: imputer: mice, fed-strategy: fedmice
    Server: fed-strategy: fedmice
    ============================================================
    
    [32m[1mImputation Start ...[0m
    


    ICE Iterations:   0%|          | 0/20 [00:00<?, ?it/s]



    Feature_idx:   0%|          | 0/14 [00:00<?, ?it/s]



    Feature_idx:   0%|          | 0/14 [00:00<?, ?it/s]



    Feature_idx:   0%|          | 0/14 [00:00<?, ?it/s]



    Feature_idx:   0%|          | 0/14 [00:00<?, ?it/s]



    Feature_idx:   0%|          | 0/14 [00:00<?, ?it/s]



    Feature_idx:   0%|          | 0/14 [00:00<?, ?it/s]



    Feature_idx:   0%|          | 0/14 [00:00<?, ?it/s]



    Feature_idx:   0%|          | 0/14 [00:00<?, ?it/s]


    [32m[1mFinished. Running time: 0.6903 seconds[0m
    

# Evaluation


```python
%load_ext autoreload
%autoreload 2
from fedimpute.evaluation import Evaluator

evaluator = Evaluator()

X_train_imps, y_trains = env.get_data(client_ids='all', data_type = 'train_imp', include_y=True)
X_tests, y_tests = env.get_data(client_ids='all', data_type = 'test', include_y=True)
X_test_imps = env.get_data(client_ids='all', data_type = 'test_imp')
X_global_test, y_global_test = env.get_data(data_type = 'global_test', include_y = True)
X_global_test_imp = env.get_data(data_type = 'global_test_imp')
data_config = env.get_data(data_type = 'config')
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    

### Federated Prediction


```python
evaluator.run_fed_regression_analysis(
    X_train_imps = X_train_imps,
    y_trains = y_trains,
    data_config = data_config
)
evaluator.show_fed_regression_results()
```

                          Federated Logit Regression Result                       
    ==============================================================================
    Dep. Variable:                    num   No. Observations:                  742
    Model:                          Logit   Df Residuals:                      727
    Method:                           MLE   Df Model:                           14
    Date:                Mon, 21 Apr 2025   Pseudo R-squ.:                  0.4122
    Time:                        16:45:26   Log-Likelihood:                -299.78
    converged:                       True   LL-Null:                       -509.99
    Covariance Type:            nonrobust   LLR p-value:                 6.246e-81
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -2.0137      0.211     -9.561      0.000      -2.426      -1.601
    age            1.2148      0.095     12.775      0.000       1.028       1.401
    trestbps       0.7584      0.205      3.694      0.000       0.356       1.161
    chol          -0.8222      0.082    -10.029      0.000      -0.983      -0.662
    thalach       -1.2272      0.122    -10.030      0.000      -1.467      -0.987
    oldpeak        4.1905      0.180     23.263      0.000       3.837       4.544
    slope          0.9589      0.081     11.897      0.000       0.801       1.117
    sex_1.0        1.2922      0.077     16.878      0.000       1.142       1.442
    cp_2.0        -0.8841      0.072    -12.272      0.000      -1.025      -0.743
    cp_3.0        -0.3322      0.060     -5.506      0.000      -0.450      -0.214
    cp_4.0         1.1759      0.060     19.538      0.000       1.058       1.294
    fbs_0.0       -1.6318      0.101    -16.154      0.000      -1.830      -1.434
    fbs_1.0       -0.6695      0.103     -6.470      0.000      -0.872      -0.467
    exang_0.0     -0.3233      0.037     -8.806      0.000      -0.395      -0.251
    exang_1.0      0.7180      0.038     19.114      0.000       0.644       0.792
    ==============================================================================
    


```python
ret = evaluator.run_fed_prediction(
    X_train_imps = X_train_imps,
    y_trains = y_trains,
    X_tests = X_test_imps,
    y_tests = y_tests,
    X_test_global = X_global_test_imp,
    y_test_global = y_global_test,
    data_config = data_config,
    model_name = 'lr',
    seed= 0
)

evaluator.show_fed_prediction_results()
```

    (149, 14) (149,)
    

                                                                    1.09it/s]

    ===============================================================
    Downstream Prediction (Fed)
    ===============================================================
     Personalized    accuracy       f1         auc         prc
    --------------  ----------  ----------  ----------  ----------
       Client 1       0.714       0.692       0.713       0.771
       Client 2       0.926       0.889       0.988       0.981
       Client 3       0.364       0.533       0.000       0.798
       Client 4       0.500       0.609       0.462       0.697
      ----------    ----------  ----------  ----------  ----------
        Global        0.809       0.804       0.891       0.903
    ===============================================================
    

    
