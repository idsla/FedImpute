import pandas as pd
import missingno
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


def visualize_missing_data(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 12), squeeze=False)
    axes[0, 0].set_title("Missing proportion")
    missingno.bar(df, fontsize=8, color='lightblue', ax=axes[0, 0])
    axes[0, 1].set_title("Missing matrix")
    missingno.matrix(df, fontsize=8, ax=axes[0, 1], sparkline=False)
    plt.tight_layout()
    plt.show()


# sklearn train and evaluate model with grid search logistic regression
def train_and_evaluate_model(
        X_train, y_train, X_test, y_test, model_name='logistic_regression', grid_search=False, scoring='accuracy',
        verbose=0
):
    if model_name == 'logistic_regression':
        model = LogisticRegression(random_state=2022)
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100, 500],
            'solver': ['liblinear', 'saga']
        }
    elif model_name == 'random_forest':
        model = RandomForestClassifier(random_state=2022)
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [5, 10, 15, 20, 25, 30],
            'min_samples_split': [2, 5, 10, 15, 100],
            'min_samples_leaf': [1, 2, 5, 10]
        }
    else:
        raise ValueError(f"Model {model_name} is not supported")

    if grid_search:
        grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    evaluation_result = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    }

    if verbose > 0:
        print(f"Accuracy: {evaluation_result['accuracy']}")
        print(f"F1: {evaluation_result['f1']}")
        print(f"ROC AUC: {evaluation_result['roc_auc']}")

    return evaluation_result