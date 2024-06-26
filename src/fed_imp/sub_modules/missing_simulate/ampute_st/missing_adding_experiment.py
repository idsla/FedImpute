from modules.data_preprocessing import load_data
from .utils import train_and_evaluate_model, visualize_missing_data
from .MissingAdder import MissingAdder
from .generate_missing_config import generate_missing_add_config
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import pandas as pd


def experiment(dataset, model_name, missing_adding_params, missing_mechanism_name, seed=0):

    # load data
    train_data, test_data, data_config = load_data(dataset)
    target = data_config['target']
    X_train, y_train = train_data.drop(target, axis=1), train_data[target]
    X_test, y_test = test_data.drop(target, axis=1), test_data[target]

    # train and evaluate complete data
    if missing_adding_params['missing_ratio'] == 0:
        evaluation_ret = train_and_evaluate_model(
            X_train, y_train, X_test, y_test, model_name=model_name, verbose=1
        )
    else:
        missing_adding_params['dataset_config'] = data_config
        missing_adding_params['seed'] = seed
        # missing parameters
        missing_config = generate_missing_add_config(missing_adding_params)

        # add missing data to training data using MultivariateAmputation
        ma = MissingAdder(seed=seed, vars_missing_config=missing_config)
        X_train_ms, y_train = ma.add_missing(X_train, y_train)

        # # visualize missing data
        visualize_missing_data(X_train_ms)
        #
        # # imputation
        imp = IterativeImputer(random_state=seed, max_iter=10, sample_posterior=True, verbose=0)
        X_train_imp = imp.fit_transform(X_train_ms)

        # # train and evaluate imputed data
        evaluation_ret = train_and_evaluate_model(
            X_train_imp, y_train, X_test, y_test, model_name=model_name, verbose=1)

    evaluation_ret['missing_ratio'] = missing_adding_params['missing_ratio']
    evaluation_ret['missing_mechanism'] = missing_mechanism_name
    evaluation_ret['n_features_missing'] = missing_adding_params['incomplete_vars']['num_to_select']

    return evaluation_ret


def run_experiments(experiment_configs):
    print("Total experiments: ", len(experiment_configs))
    print('Running experiments...')
    rets = []
    for idx, experiment_config in enumerate(experiment_configs):
        print("=================================================================================")
        print('Experiment: ', idx + 1)
        print("Missing ratio: {} | Missing mechanism: {} | Number of features missing: {}".format(
            experiment_config['missing_adding_params']['missing_ratio'],
            experiment_config['missing_mechanism_name'],
            experiment_config['missing_adding_params']['incomplete_vars']['num_to_select'])
        )
        ret = experiment(**experiment_config)
        rets.append(ret)
    print('Done!')
    return rets


def result_analysis(rets, x):
    x = [ret[x] for ret in rets]
    accuracy = [ret['accuracy'] for ret in rets]
    f1 = [ret['f1'] for ret in rets]
    auc = [ret['auc'] for ret in rets]
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(accuracy, label='accuracy')
    ax.plot(f1, label='f1')
    ax.plot(auc, label='auc')
    ax.legend()
    plt.show()


def result_analysis_table(rets, filename='results.csv'):
    df = pd.DataFrame(rets)
    df.to_csv(filename, index=False)
    return df


