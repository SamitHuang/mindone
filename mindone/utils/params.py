import copy
import os
import re
from typing import List, Optional, Union

from mindcv.utils.download import DownLoad

import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter
from mindspore import log as logger

# from mindspore._checkparam import Validator
from mindspore.train.serialization import _load_dismatch_prefix_params, _update_param


def is_url(string):
    # Regex to check for URL patterns
    url_pattern = re.compile(r"^(http|https|ftp)://")
    return bool(url_pattern.match(string))


def load_param_into_net_with_filter(
    net: nn.Cell, parameter_dict: dict, strict_load: bool = False, filter: Optional[List] = None
):
    """
    Load parameters into network, return parameter list that are not loaded in the network.

    Args:
        net (Cell): The network where the parameters will be loaded.
        parameter_dict (dict): The dictionary generated by load checkpoint file,
                               it is a dictionary consisting of key: parameters's name, value: parameter.
        strict_load (bool): Whether to strict load the parameter into net. If False, it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: False.
        filter (List): If not None, it will only load the parameters in the given list. Default: None.

    Returns:
        param_not_load (List), the parameter name in model which are not loaded into the network.
        ckpt_not_load (List), the parameter name in checkpoint file which are not loaded into the network.

    Raises:
        TypeError: Argument is not a Cell, or parameter_dict is not a Parameter dictionary.

    Examples:
        >>> import mindspore as ms
        >>>
        >>> net = Net()
        >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
        >>> param_dict = ms.load_checkpoint(ckpt_file_name, filter_prefix="conv1")
        >>> param_not_load, _ = ms.load_param_into_net(net, param_dict)
        >>> print(param_not_load)
        ['conv1.weight']
    """
    if not isinstance(net, nn.Cell):
        logger.critical("Failed to combine the net and the parameters.")
        msg = "For 'load_param_into_net', the argument 'net' should be a Cell, but got {}.".format(type(net))
        raise TypeError(msg)

    if not isinstance(parameter_dict, dict):
        logger.critical("Failed to combine the net and the parameters.")
        msg = "For 'load_param_into_net', the argument 'parameter_dict' should be a dict, " "but got {}.".format(
            type(parameter_dict)
        )
        raise TypeError(msg)
    for key, value in parameter_dict.items():
        if not isinstance(key, str) or not isinstance(value, (Parameter, str)):
            logger.critical("Load parameters into net failed.")
            msg = (
                "For 'parameter_dict', the element in the argument 'parameter_dict' should be a "
                "'str' and 'Parameter' , but got {} and {}.".format(type(key), type(value))
            )
            raise TypeError(msg)

    # TODO: replace by otherway to do check_bool
    # strict_load = Validator.check_bool(strict_load)
    logger.info("Execute the process of loading parameters into net.")
    net.init_parameters_data()
    param_not_load = []
    ckpt_not_load = list(parameter_dict.keys())
    for _, param in net.parameters_and_names():
        if param.name in parameter_dict:
            new_param = copy.deepcopy(parameter_dict[param.name])
            _update_param(param, new_param, strict_load)
            ckpt_not_load.remove(param.name)
        else:
            param_not_load.append(param.name)

    if param_not_load and not strict_load:
        _load_dismatch_prefix_params(net, parameter_dict, param_not_load, strict_load)

    logger.info("Loading parameters into net is finished.")
    if filter:
        param_all_load_flag = len(set(param_not_load).intersection(set(filter))) == 0
        if param_all_load_flag:
            param_not_load.clear()
    if param_not_load:
        logger.warning(
            "For 'load_param_into_net', "
            "{} parameters in the 'net' are not loaded, because they are not in the "
            "'parameter_dict', please check whether the network structure is consistent "
            "when training and loading checkpoint.".format(len(param_not_load))
        )
        for param_name in param_not_load:
            logger.warning("{} is not loaded.".format(param_name))
    return param_not_load, ckpt_not_load


def load_from_pretrained(
    net: nn.Cell,
    checkpoint: Union[str, dict],
    ignore_net_params_not_loaded=False,
    ensure_all_ckpt_params_loaded=False,
    cache_dir: str = None,
):
    """load checkpoint into network.

    Args:
        net: network
        checkpoint: local file path to checkpoint, or url to download checkpoint, or a dict for network parameters
        ignore_net_params_not_loaded: set True for inference if only a part of network needs to be loaded, the flushing net-not-loaded warnings will disappear.
        ensure_all_ckpt_params_loaded : set True for inference if you want to ensure no checkpoint param is missed in loading
        cache_dir: directory to cache the downloaded checkpoint, only effective when `checkpoint` is a url.
    """
    if isinstance(checkpoint, str):
        if is_url(checkpoint):
            url = checkpoint
            cache_dir = os.path.join(os.path.expanduser("~"), ".mindspore/models") if cache_dir is None else cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            DownLoad().download_url(url, path=cache_dir)
            checkpoint = os.path.join(cache_dir, os.path.basename(url))
        if os.path.exists(checkpoint):
            param_dict = ms.load_checkpoint(checkpoint)
        else:
            raise FileNotFoundError(f"{checkpoint} doesn't exist")
    elif isinstance(checkpoint, dict):
        param_dict = checkpoint
    else:
        raise TypeError(f"unknown checkpoint type: {checkpoint}")

    if param_dict:
        if ignore_net_params_not_loaded:
            filter = param_dict.keys()
        else:
            filter = None
        param_not_load, ckpt_not_load = load_param_into_net_with_filter(net, param_dict, filter=filter)

        if ensure_all_ckpt_params_loaded:
            assert (
                len(ckpt_not_load) == 0
            ), f"All params in checkpoint must be loaded. but got these not loaded {ckpt_not_load}"

        if not ignore_net_params_not_loaded:
            if len(param_not_load) > 0:
                logger.info("Net params not loaded: {}".format([p for p in param_not_load if not p.startswith("adam")]))
        logger.info("Checkpoint params not loaded: {}".format([p for p in ckpt_not_load if not p.startswith("adam")]))


def count_params(model, verbose=False):
    total_params = sum([param.size for param in model.get_parameters()])
    trainable_params = sum([param.size for param in model.get_parameters() if param.requires_grad])

    if verbose:
        logger.info(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params, trainable_params
