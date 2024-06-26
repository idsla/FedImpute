import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno
import numpy as np
from src.fed_imp.sub_modules.server.load_server import load_server
from src.fed_imp.sub_modules.client.simple_client import SimpleClient
from src.modules.data_partition import data_partition
from src.modules.data_spliting import split_train_test
from src.modules.data_preprocessing import load_data
import random
from src.fed_imp.sub_modules.missing_simulate.missing_adder_new import add_missing
from src.fed_imp.sub_modules.client.client_factory import ClientsFactory
from src.fed_imp.sub_modules.strategy.strategy_imp import StrategyImputation
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN

def run_simulation(configuration, clients, test_data, seed, vis = True):
    # Create Imputation Strategy
    imp_strategy = configuration['agg_strategy_imp']['strategy']
    params = configuration['algo_params'][imp_strategy]

    strategy_imp = StrategyImputation(strategy=imp_strategy, params=params)

    # Create Server
    server_type = configuration['server_type']
    server_config = configuration['server']
    server_config["n_cols"] = test_data.shape[1] - 1

    pred_config = configuration['pred_model']
    pred_config['model_params']['input_feature_dim'] = test_data.shape[1] - 1
    pred_config['model_params']['output_classes_dim'] = len(np.unique(test_data.iloc[:, -1].values))

    server = load_server(
        server_type,
        clients=clients,
        strategy_imp=strategy_imp,
        server_config=server_config,
        pred_config=pred_config,
        test_data=test_data.values,
        seed=seed,
        track=configuration['track'],
        run_prediction=configuration['prediction'],
        persist_data=configuration['save_state'],
    )

    # return server
    ret00 = server.run()
    if vis:
        vis_imp(ret00)
    print(ret00['imp_result'])
    #sklearn_evaluation(ret00)

    return server, ret00

def simulate_scenario(configuration):
    num_clients = configuration['num_clients']
    # set random seed
    seed = configuration['experiment']['seed']
    mtp = configuration['experiment']['mtp']
    tune_params = configuration['tune_params']
    random.seed(seed)  # seed for split data

    # load data
    dataset_params = configuration['data']
    data, data_config = load_data(**dataset_params)
    regression = data_config['task_type'] == 'regression'

    n_rounds = configuration['experiment']['n_rounds']
    test_size = configuration['experiment'].get('test_size', 0.1)
    if n_rounds == 1:
        n_rounds_data = split_train_test(data, n_folds=2, seed=seed, test_size=test_size, regression=regression)
    else:
        n_rounds_data = split_train_test(
            data, n_folds=n_rounds, seed=seed, test_size=test_size, regression=regression
            )

    # n rounds average
    train_data, test_data = n_rounds_data[0]
    print(train_data.shape, test_data.shape)

    imbalance_strategy = configuration.get('handle_imbalance', None)
    if imbalance_strategy == 'oversampling':
        columns = train_data.columns
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        ros = RandomOverSampler(random_state=seed)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        train_data = pd.DataFrame(np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1), columns=columns)
    elif imbalance_strategy == 'smote':
        columns = train_data.columns
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        smote = SMOTE(random_state=seed, n_jobs=-1)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        train_data = pd.DataFrame(np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1), columns=columns)
    elif imbalance_strategy == 'adasyn':
        columns = train_data.columns
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        ada = ADASYN(random_state=seed, n_jobs=-1)
        X_train, y_train = ada.fit_resample(X_train, y_train)
        train_data = pd.DataFrame(np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1), columns=columns)

    print(train_data.shape)

    new_seed = (seed + 10087 * 0) % (2 ^ 23)
    regression = data_config['task_type'] == 'regression'
    data_partition_params = configuration['data_partition']
    data_partitions = data_partition(
        **data_partition_params, data=train_data.values, n_clients=num_clients, seed=new_seed,
        regression=regression
    )
    
    print("sample sizes: ", [item.shape[0] for item in data_partitions])

    missing_params = configuration['missing_simulate']
    cols = np.arange(0, train_data.shape[1] - 1)
    scenario = missing_params
    data_ms_clients2 = add_missing(
        train_data_list=data_partitions, scenario=scenario, cols=cols, seed=new_seed
    )

    client_factory = ClientsFactory(debug=False)
    clients = client_factory.generate_clients(
        num_clients, data_partitions, data_ms_clients2, test_data.values, data_config,
        configuration['imputation'], seed=new_seed
    )
    #visualize_ms(data_ms_clients2)

    return clients, test_data, new_seed

def NN_evaluation(ret0, type='centralized', n_rounds = 500, server_config_tmpl = None, imbalance = None):
    clients_ = {}
    split_indices = ret0['data']['split_indices'].tolist()
    data_imp = np.split(ret0['data']['imputed_data'], split_indices)
    missing_mask = np.split(ret0['data']['missing_mask'], split_indices)
    data_true = np.split(ret0['data']['origin_data'], split_indices)
    n_clients = len(data_imp)
    test_data = ret0['data']['test_data']
    split_indices = ret0['data']['split_indices']
    for client_id in range(n_clients):
        clients_[client_id] = SimpleClient(
            client_id=client_id,
            data_imp=data_imp[client_id],
            missing_mask=missing_mask[client_id],
            data_true=data_true[client_id],
            data_test=test_data,
            imbalance=imbalance,
        )

    pred_config = server_config_tmpl.copy()
    if type == 'centralized':
        server_name = 'central_mlp_pytorch_pred'
    elif type == 'fedavg':
        server_name = 'fedavg_mlp_pytorch_pred'
    else:
        raise ValueError('type should be centralized or fedavg')
    server_pred_config = pred_config['server_pred_config']
    server_pred_config['train_params']['pred_round'] = n_rounds
    server_config = pred_config['server_config']
    server_ = load_server(
            server_name, clients=clients_, server_config=server_config, pred_config=server_pred_config,
            test_data=test_data
        )

    pred_ret2 = server_.prediction()
    print(pred_ret2["accu_mean"], pred_ret2['f1_mean'], pred_ret2['roc_auc_mean'], pred_ret2['prc_auc_mean'])
    return pred_ret2

def visualize_ms(clients_ms_datas:list, sort_patterns: bool = False):
    n_cols = 5
    n_clients = len(clients_ms_datas)
    n_rows = (n_clients + 4)//n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3), squeeze=False)
    for i in range(n_clients):
        ax = axes[i//n_cols, i%n_cols]
        d = pd.DataFrame(clients_ms_datas[i])
        d = d.sort_values(by=d.columns[-1])
        if sort_patterns:
            msno.matrix(d, ax=ax, sparkline=False, sort='ascending')
        else:
            msno.matrix(d, ax=ax, sparkline=False)
        ax.set_title('Client {}'.format(i))
    plt.tight_layout()

# def correlation(original_data, centralized_data):
#     original_df = pd.DataFrame(original_data)
#     target_col = original_df.columns[-1]
#     correlation_ret = original_df.corrwith(original_df[target_col], method=correlation_ratio).sort_values(ascending=False)
#     print(correlation_ret)

#     centralized_df = pd.DataFrame(centralized_data)
#     target_col = centralized_df.columns[-1]
#     correlation_ret = centralized_df.corrwith(original_df[target_col], method=correlation_ratio).sort_values(ascending=False)
#     print(correlation_ret)

def run_pred(clf_name, X_train, y_train, X_test, y_test):
    accus = []
    for i in range(5):
        seed = 21 + i*93940
        if clf_name == 'LR':
            clf = LogisticRegression(random_state=seed, max_iter=1000)
        elif clf_name == 'MLP':
            clf = MLPClassifier(
                [32, 32], batch_size = 128, random_state=seed, alpha = 0.001, max_iter = 1000)
        else:
            raise ValueError('clf_name should be LR or MLP')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accus.append(metrics.accuracy_score(y_test, y_pred))
    return np.mean(accus), np.std(accus)

def sklearn_evaluation(rets):

    original_data = rets['data']['origin_data'].reshape(-1, rets['data']['origin_data'].shape[-1])
    centralized_data = rets['data']['imputed_data'].reshape(-1, rets['data']['origin_data'].shape[-1])
    test_data = rets['data']['test_data']

    X_train = centralized_data[:, :-1]
    y_train = centralized_data[:, -1]
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    accu, std = run_pred('MLP', X_train, y_train, X_test, y_test)
    print("Accuracy imputed centralized MLP:{:.4f} ({:.3f})".format(accu, std))

    accu, std = run_pred('LR', X_train, y_train, X_test, y_test)
    print("Accuracy imputed centralized LR:{:.4f} ({:.3f})".format(accu, std))

    # X_train = original_data[:, :-1]
    # y_train = original_data[:, -1]
    # X_test = test_data[:, :-1]
    # y_test = test_data[:, -1]
    # accu, std = run_pred('MLP', X_train, y_train, X_test, y_test)
    # print("Accuracy orignal centralized MLP:{:.4f} ({:.3f})".format(accu, std))

    # accu, std = run_pred('LR', X_train, y_train, X_test, y_test)
    # print("Accuracy orignal centralized MLP:{:.4f} ({:.3f})".format(accu, std))


def vis_imp(ret):
    x = list(range(len(ret['client_imp_history'])))
    client_ids = list(ret['client_imp_history'][0][2]['metrics'].keys())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for client_id in client_ids:
        for idx, metric in enumerate(['imp@rmse', 'imp@w2', 'imp@sliced_ws']):
            y = [ret['client_imp_history'][i][2]['metrics'][client_id][metric] for i in x]
            ax[idx].plot(x, y, label=client_id)
            ax[idx].set_title(metric)
    
    if len(client_ids) < 20:
        ax[-1].legend(loc='upper right',
                    bbox_to_anchor=(1.1, 1.1), ncol=4, fontsize=10)
        
    plt.tight_layout()
    plt.show()