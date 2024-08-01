from typing import Union, List, Dict, Tuple
import torch
import numpy as np
from collections import OrderedDict


def get_parameters(
        model: torch.nn.Module,
        trainable_only: bool = False,
        return_type: str = 'state_dict',
        copy: bool = True
) -> Union[Dict[str, np.ndarray], List[torch.Tensor], List[Tuple[str, torch.Tensor]]]:
    """
    Collect parameters from a PyTorch model and return them in the specified format.

    Args:
        model (torch.nn.Module): The PyTorch model containing parameters.
        trainable_only (bool): If True, only return parameters with requires_grad=True. Default is False.
        return_type (str): Specifies the return type. 'state_dict' for an ordered dictionary of tensors, 'numpy_dict' for a
                            dictionary of numpy arrays, 'named_parameters' for a list of named torch tensors.
                           Default is 'state_dict'.

    Returns:
        Union[Dict[str, np.ndarray], List[torch.Tensor]], List[Tuple[str, torch.Tensor]]:
        Parameters in the specified format.
        :param return_type:
        :param trainable_only:
        :param model:
        :param copy:
    """

    if return_type not in ['state_dict', 'numpy_dict', 'named_parameters', 'parameters']:
        raise ValueError("return_type must be either 'state_dict' or 'numpy_dict' or 'named_parameters' or 'parameters'")

    if copy:

        if return_type == 'numpy_dict':
            result = {}
            if not trainable_only:
                for name, param in model.state_dict(keep_vars=True).items():
                    result[name] = param.detach().cpu().numpy().copy()
            else:
                for name, param in model.state_dict(keep_vars=True).items():
                    if isinstance(param, torch.Tensor) and param.requires_grad:
                        result[name] = param.detach().cpu().numpy().copy()
            return result
        elif return_type == 'state_dict':
            if not trainable_only:
                return OrderedDict((name, param.detach().clone()) for name, param in model.named_parameters())
            else:
                return (
                    OrderedDict((name, param.detach().clone()) for name, param in model.named_parameters()
                                if param.requires_grad)
                )
        elif return_type == 'parameters':

            if not trainable_only:
                return [param.detach().clone() for param in model.parameters()]
            else:
                return [param.detach().clone() for param in model.parameters() if param.requires_grad]

        else:
            if not trainable_only:
                return [(name, param.detach().clone()) for name, param in model.named_parameters()]
            else:
                return [(name, param.detach().clone()) for name, param in model.named_parameters() if param.requires_grad]

    else:

        if return_type == 'numpy_dict' or return_type == 'state_dict':
            raise ValueError("copy=False is only supported for 'parameters' and 'named_parameters'")
        elif return_type == 'parameters':
            if not trainable_only:
                return [param for param in model.parameters()]
            else:
                return [param for param in model.parameters() if param.requires_grad]
        else:
            if not trainable_only:
                return [(name, param) for name, param in model.named_parameters()]
            else:
                return [(name, param) for name, param in model.named_parameters() if
                        param.requires_grad]


def convert_params_format(
        params: Union[OrderedDict, Dict[str, np.ndarray], List[Tuple[str, torch.Tensor]]],
        output_type: str = 'state_dict'
) -> Union[OrderedDict, Dict[str, np.ndarray], List[torch.Tensor], List[Tuple[str, torch.Tensor]]]:
    """
    Convert between different parameter representations: state_dict, dict of numpy arrays, and list of named tensors.

    Args:
        params: Input parameters in one of the following formats:
                - OrderedDict (state_dict)
                - Dict[str, np.ndarray] (dict of numpy arrays)
                - List[Tuple[str, torch.Tensor]] (named_parameters)
        output_type: Desired output type. Options are 'state_dict', 'numpy_dict',
                     'parameters', or 'named_parameters'.

    Returns:
        Parameters in the specified output format.

    Raises:
        ValueError: If an invalid output_type is specified or if the input format is not recognized.
    """
    if output_type not in ['state_dict', 'numpy_dict', 'parameters', 'named_parameters']:
        raise ValueError("output_type must be 'state_dict', 'numpy_dict', 'parameters', or 'named_parameters'")

    # Determine input type
    if isinstance(params, OrderedDict):
        input_type = 'state_dict'
    elif isinstance(params, dict) and all(isinstance(v, np.ndarray) for v in params.values()):
        input_type = 'numpy_dict'
    elif (
        isinstance(params, list) and all(isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str) and
                                         isinstance(p[1], torch.Tensor) for p in params)
    ):
        input_type = 'named_parameters'
    else:
        raise ValueError("Input format not recognized")

    # Convert to state_dict if necessary
    if input_type == 'named_parameters':
        state_dict = OrderedDict(params)
    elif input_type == 'numpy_dict':
        state_dict = OrderedDict((k, torch.from_numpy(v)) for k, v in params.items())
    else:  # input_type == 'state_dict'
        state_dict = params

    # Convert to desired output type
    if output_type == 'state_dict':
        return state_dict
    elif output_type == 'numpy_dict':
        return {k: v.cpu().numpy() for k, v in state_dict.items()}
    elif output_type == 'parameters':
        return list(state_dict.values())
    else:  # output_type == 'named_parameters'
        return list(state_dict.items())
