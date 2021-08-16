def set_requires_grad(model, val):
    for param in model.parameters():
        param.requires_grad = val


def ema(target_param, online_param, alpha):
    if alpha is None:
        return online_param
    return alpha * target_param + (1 - alpha) * online_param
