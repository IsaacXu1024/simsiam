def set_requires_grad(model, val):
    """
    Set pytorch model's require grad property to val.

    Copyright (c) 2020 Phil Wang Redistributed under the MIT license.
    Function taken from: https://github.com/lucidrains/byol-pytorch
    """
    for param in model.parameters():
        param.requires_grad = val


def ema(target_param, online_param, alpha):
    if alpha is None:
        return online_param
    return alpha * target_param + (1 - alpha) * online_param
