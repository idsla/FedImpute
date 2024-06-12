import torch
from collections import OrderedDict
from typing import Union, List, Tuple


def trainable_params(
        src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module],
        detach=False,
        requires_name=False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[str]]]:
    """Collect all parameters in `src` that `.requires_grad = True` into a list and return it.

    Args:
        src (Union[OrderedDict[str, torch.Tensor], torch.nn.Module]): The source that contains parameters.
        requires_name (bool, optional): If set to `True`, The names of parameters would also return in another list.
            Defaults to False.
        detach (bool, optional): If set to `True`, the list would contain `param.detach().clone()` rather than `param`.
            Defaults to False.

    Returns:
        List of parameters [, names of parameters].
    """
    func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
    parameters = []
    keys = []

    # model.parameters() returns a generator of parameters
    if isinstance(src, OrderedDict):
        for name, param in src.items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)
    # model class - nn.Module
    elif isinstance(src, torch.nn.Module):
        for name, param in src.state_dict(keep_vars=True).items():
            if param.requires_grad:
                parameters.append(func(param))
                keys.append(name)

    if requires_name:
        return parameters, keys
    else:
        return parameters
