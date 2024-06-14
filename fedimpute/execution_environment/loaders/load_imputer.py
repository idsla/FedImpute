from ..imputation.imputers import (
    LinearICEImputer,
    SimpleImputer, EMImputer,
    ICEGradImputer, GAINImputer, MIWAEImputer, MissForestImputer,
)


def load_imputer(name, imputer_params):

    ####################################################################################################################
    # Traditional ML Imputation Models
    if name == 'mean':
        return SimpleImputer(**imputer_params)
    elif name == 'ice':
        return LinearICEImputer(**imputer_params)
    # elif name == 'linear_sgd_ice':
    #     return ICEGradImputer(**imputer_params)
    elif name == 'em':
        return EMImputer(**imputer_params)
    elif name == 'missforest':
        return MissForestImputer(**imputer_params)

    ####################################################################################################################
    # Deep Learning NN Imputation Models
    # elif name == 'mlp_ice':
    #     return ICEGradImputer(**imputer_params)
    elif name == 'gain':
        return GAINImputer(**imputer_params)
    elif name == 'miwae':
        return MIWAEImputer(name='miwae', **imputer_params)
    # elif name == 'notmiwae':
    #     return MIWAEImputer(name='notmiwae', **imputer_params)
    else:
        raise NotImplementedError
