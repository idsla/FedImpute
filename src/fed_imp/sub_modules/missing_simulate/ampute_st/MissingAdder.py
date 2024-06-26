import pandas as pd
from modules.missing_simulate.ampute_st.missing_adding_utils import (
    add_missing_single_column_mcar, add_missing_single_column_mar, MissingParamsVar
)
import random as random
from typing import Union, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class VarMissingConfig:
    missing_mechanism: str
    missing_ratio: float
    associated_vars: List[int] = None
    associated_vars_range: Union[List, Tuple[float, float]] = None


class MissingAdder:

    def __init__(
            self,
            vars_missing_config: Dict[int, Dict],
            seed=21
    ):

        # missing options and parameters
        incomplete_vars = list(vars_missing_config.keys())
        self.missing_configs = {}
        for var in incomplete_vars:
            self.missing_configs[var] = VarMissingConfig(**vars_missing_config[var])

        self.seed = seed

    def add_missing(self, X_train, y_train):

        X_train_original = X_train.copy()
        y_train_original = y_train.copy()

        data = pd.concat([X_train_original, y_train_original], axis=1).reset_index(drop=True)

        # add missing value to each of features
        for feature_idx, missing_config in self.missing_configs.items():
            missing_mechanism = missing_config.missing_mechanism
            missing_ratio = missing_config.missing_ratio

            # loading other parameters
            if missing_mechanism == 'mcar':
                # add missing value
                data.iloc[:, feature_idx] = add_missing_single_column_mcar(
                    data.iloc[:, feature_idx], missing_ratio, self.seed + feature_idx
                )
            elif missing_mechanism == 'mar' or missing_mechanism == 'mnar':
                if missing_config.associated_vars is None:
                    raise ValueError('associated_vars should be provided for MAR')

                associated_vars = []
                assciated_vars_idx = []
                for var, var_range in zip(missing_config.associated_vars, missing_config.associated_vars_range):
                    if isinstance(var_range, list):
                        associated_vars.append(
                            MissingParamsVar(
                                var_idx=var, var_type='cat', var_filter_range=var_range
                            )
                        )
                    elif isinstance(var_range, tuple):
                        associated_vars.append(
                            MissingParamsVar(
                                var_idx=var, var_type='num', var_filter_range=var_range
                            )
                        )
                    else:
                        raise TypeError('associated_percentiles should be either string or tuple')

                    assciated_vars_idx.append(var)

                # add missing value
                data.iloc[:, feature_idx] = add_missing_single_column_mar(
                    data.iloc[:, feature_idx], data.iloc[:, assciated_vars_idx], associated_vars, missing_ratio,
                    self.seed + feature_idx
                )
            # elif missing_mechanism == 'mnar':
            #     associated_percentiles = missing_config.associated_percentiles
            #     if isinstance(associated_percentiles, str):
            #         lower_percentile, upper_percentile = self.get_percentiles_from_strategy(associated_percentiles)
            #     elif isinstance(associated_percentiles, tuple):
            #         lower_percentile, upper_percentile = associated_percentiles
            #     else:
            #         raise TypeError('associated_percentiles should be either string or tuple')
            #
            #     # add missing value
            #     data.iloc[:, feature_idx] = add_missing_single_column_mnar(
            #         data.iloc[:, feature_idx], 'num', (lower_percentile, upper_percentile), missing_ratio, self.seed
            #     )
            else:
                raise ValueError('missing_mechanism should be either mcar, mar, or mnar')

        return data.drop(columns=[data.columns[-1]]), data[data.columns[-1]]

    # get percentiles from strategies
    def get_percentiles_from_strategy(self, strategy):
        if strategy == 'random':
            seed = self.seed
            random.seed(seed)
            lower_percentile = random.uniform(0, 1)
            upper_percentile = lower_percentile + 0.5
            if upper_percentile > 1:
                upper_percentile = upper_percentile - 1
            return lower_percentile, upper_percentile
        elif strategy == 'upper':
            return 0.5, 1
        elif strategy == 'lower':
            return 0, 0.5
