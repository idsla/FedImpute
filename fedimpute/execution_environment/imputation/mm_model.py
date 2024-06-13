from abc import ABC
from random import seed

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier


# Mechanism model
# 1. feature specific1: input: x (x1, ..., xd) -> output: mask (m1)
# 2. feature specific2: input: z (z1, ..., zl) -> output: mask (m1)
# 2. joint1: input: x (x1, ..., xd) -> output: mask (m1, ..., md)
# 3. joint2: input: z (z1, ..., zl) -> output: mask (m1, ..., md)
# mm_model_type: linear, neural network
########################################################################################################################
# Mechanism Model Loader
########################################################################################################################
def mechanism_model_loader(mm_model_name, model_type):
    if model_type == 'fs':
        if mm_model_name == 'self-masking':
            pass
        elif mm_model_name == 'linear':
            pass
        elif mm_model_name == 'mlp':
            pass
        else:
            raise ValueError('Unknown mechanism model name: {}'.format(mm_model_name))
    elif model_type == 'jm':
        if mm_model_name == 'self-masking':
            pass
        elif mm_model_name == 'linear':
            pass
        elif mm_model_name == 'mlp':
            pass
    else:
        raise ValueError('Unknown mechanism model type: {}'.format(model_type))


########################################################################################################################
class BaseMechanismModel(ABC):

    def __init__(self):
        pass


########################################################################################################################
# Mechanism Model
class SelfMaskModel(BaseMechanismModel):

    def __init__(self, model_type, mm_model_params, seed=0):
        super().__init__()

        self.model_type = model_type
        self.mm_model_params = mm_model_params
        if model_type == 'fs':
            pass
        elif model_type == 'jm':
            pass

    def fit(self, X, y):
        self.mm_model.fit(X, y)


class LogitMechanismModel(BaseMechanismModel):

    def __init__(self, model_type, mm_model_params, seed=0):
        super().__init__()

        self.model_type = model_type
        self.mm_model_params = mm_model_params
        if model_type == 'fs':
            self.mm_model = LogisticRegressionCV(
                Cs=self.mm_model_params['Cs'], class_weight=self.mm_model_params['class_weight'],
                cv=StratifiedKFold(self.mm_model_params['cv']), random_state=seed, max_iter=1000, n_jobs=-1
            )
        elif model_type == 'jm':
            pass

    def fit(self, X, y):
        self.mm_model.fit(X, y)


class NNMechanismModel(BaseMechanismModel):

    def __init__(self, model_type, mm_model_params, seed=0):
        super().__init__()

        self.model_type = model_type
        self.mm_model_params = mm_model_params
        if model_type == 'fs':
            pass
        elif model_type == 'jm':
            pass

    def fit(self, X, y):
        self.mm_model.fit(X, y)