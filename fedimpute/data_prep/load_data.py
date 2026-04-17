from importlib import resources

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler


def _remove_iqr_outliers(data: pd.DataFrame, column: str) -> pd.DataFrame:
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return data.copy()

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[data[column].between(lower_bound, upper_bound)].copy()


def _convert_gaussian(data: pd.DataFrame, column: str) -> pd.DataFrame:
    data = data.copy()
    transformer = PowerTransformer(method="yeo-johnson", standardize=False)
    data[[column]] = transformer.fit_transform(data[[column]])
    return data


def load_data(data_name: str):
    """Load one of the built-in example datasets.

    Parameters
    ----------
    data_name : str
        Dataset name. Supported values are ``"codrna"``,
        ``"fed_heart_disease"``, and ``"california"``.

    Returns
    -------
    data : pandas.DataFrame or list[pandas.DataFrame]
        Prepared data with feature columns followed by the target column. For
        non-federated datasets this is a single dataframe. For naturally
        partitioned datasets such as ``"fed_heart_disease"``, this is a list of
        client dataframes in the site order used by the source data.
    data_config : dict
        Metadata used by FedImpute. It includes ``target`` (target column name),
        ``task_type`` (``"classification"`` or ``"regression"``),
        ``natural_partition`` (whether ``data`` is already split by client), and
        ``num_cols`` (number of leading numerical feature columns).
    """

    if data_name == "codrna":

        features, labels = fetch_openml(data_id=351, as_frame="auto", return_X_y=True)
        df_pred = pd.DataFrame.sparse.from_spmatrix(features).sparse.to_dense()
        df_pred.columns = [f"X{i+1}" for i in range(df_pred.shape[1])]
        df_label = pd.DataFrame(labels)
        df_label = pd.factorize(df_label[0])[0]
        df_label = pd.DataFrame(df_label, columns=["y"]).astype(int)
        data_standard = StandardScaler().fit_transform(df_pred.values)
        data_minmax = MinMaxScaler().fit_transform(data_standard)
        data = pd.DataFrame(data_minmax, columns=df_pred.columns)
        data = pd.concat([data, df_label], axis=1)
        data_config = {
            "target": "y",
            "task_type": "classification",
            "natural_partition": False,
            "num_cols": data.shape[1] - 1,
        }

        data = data.sample(n=5000, random_state=42).reset_index(drop=True)

        return data, data_config

    elif data_name == "fed_heart_disease":
        heart_disease_sites = ["cleveland", "hungarian", "switzerland", "va"]
        heart_disease_columns = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "num",
        ]

        def read_heart_disease_site(site: str) -> pd.DataFrame:
            data_file = resources.files("fedimpute.data_prep").joinpath(
                "example_data", "heart_disease", f"processed.{site}.data"
            )
            if not data_file.is_file():
                raise FileNotFoundError(
                    f"Bundled heart disease data file not found: {data_file}"
                )

            with data_file.open("rb") as file:
                site_data = pd.read_csv(file, header=None, na_values="?")
            site_data.columns = heart_disease_columns
            return site_data

        dfs = [read_heart_disease_site(site) for site in heart_disease_sites]

        df = pd.concat(dfs, axis=0).reset_index(drop=True)
        split_indices = np.cumsum([0] + [df_sub.shape[0] for df_sub in dfs])

        cat_cols = ["sex", "cp", "fbs", "exang"]
        num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "slope"]
        drop_cols = ["ca"]
        target_col = "num"

        df = df.drop(columns=drop_cols)
        df_features = df[num_cols + cat_cols].copy()
        for col in cat_cols:
            df_features[col] = df_features[col].fillna(-1)

        df_features = pd.get_dummies(df_features, columns=cat_cols, drop_first=True)
        df_target = df[target_col].copy()
        df_target = df_target.apply(lambda x: 0 if x == 0 else 1)

        scaler = StandardScaler()
        df_features[num_cols] = scaler.fit_transform(df_features[num_cols])
        scaler = MinMaxScaler()
        df_features[num_cols] = scaler.fit_transform(df_features[num_cols])

        data = pd.concat([df_features, df_target], axis=1)

        data_config = {
            "target": target_col,
            "task_type": "classification",
            "natural_partition": True,
            "num_cols": len(num_cols),
        }

        dfs = [
            data.iloc[split_indices[i] : split_indices[i + 1]]
            .reset_index(drop=True)
            .copy()
            for i in range(len(split_indices) - 1)
        ]

        return dfs, data_config

    elif data_name == "california":

        housing = fetch_california_housing()
        data = pd.DataFrame(data=housing.data, columns=housing.feature_names)
        target_col = "MedHouseVal"
        data[target_col] = housing.target

        # drop missing values
        data = data.dropna()

        # remove outliers
        data = _remove_iqr_outliers(data, "AveRooms")
        data = _remove_iqr_outliers(data, "AveBedrms")
        data = _remove_iqr_outliers(data, "Population")
        data = _remove_iqr_outliers(data, "AveOccup")

        # gaussian transform
        data = _convert_gaussian(data, "MedInc")

        num_cols = data.columns.tolist()[:-1]

        scaler = Pipeline([("standard", StandardScaler()), ("minmax", MinMaxScaler())])

        data[num_cols] = scaler.fit_transform(data[num_cols])

        data_config = {
            "target": target_col,
            "task_type": "regression",
            "natural_partition": False,
            "num_cols": len(num_cols),
        }

        sample_size = min(5000, data.shape[0])
        data = data.sample(n=sample_size, random_state=42).reset_index(drop=True)

        return data, data_config
    else:
        raise ValueError(f"Data {data_name} not found")
