# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch
from torch import nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.

    Parameters
    ----------
    base_encoder : torch.nn.Module
        Base encoder model.
    dim : int, default=2048
        Feature dimension
    pred_dim : int, default=512
        Hidden dimension of the predictor
    """

    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False),
        )  # output layer
        # hack: not use bias as it is followed by BN
        self.encoder.fc[6].bias.requires_grad = False

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim),
        )  # output layer

    def forward(self, x1, x2):
        """
        Forward step.

        Parameters
        ----------
        x1 : torch.Tensor
            First view of images.
        x2 : torch.Tensor
            Second view of images.

        Return
        ------
        p1, p2, z1, z2 :
            predictors and targets of the network

        Note
        ----
        See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()


class BYOL(SimSiam):
    """
    Build a Bootstrap Your Own Latent (BYOL) model.

    This is an implementation of `Bootstrap Your Own Latent <BYOL_>`.
    It is structured similarly to the SimSiam model, except that the target
    network is an exponential moving average of the online encoder network.

    .. _BYOL: https://arxiv.org/abs/2006.07733

    Parameters
    ----------
    base_encoder : torch.nn.Module
        Base encoder model.
    dim : int, default=2048
        Feature dimension
    pred_dim : int, default=512
        Hidden dimension of the predictor
    init_target_from_online : bool, default=False
        Whether to initialize the target network as having the same weights as
        the encoder. Default behaviour is to use a random initialization.
    alpha : float or None, default=0.99
        Weight term for exponential moving average. Larger values will cause
        the network parameters to update slower.
        If ``alpha`` is ``None``, EMA will be disabled and the target network
        will be a mirror of the encoder instead.
    """

    def __init__(
        self,
        base_encoder,
        dim=2048,
        pred_dim=512,
        init_target_from_online=False,
        alpha=0.99,
    ):
        super(BYOL, self).__init__(base_encoder, dim, pred_dim)

        self.alpha = alpha

        # build target model
        self.target_encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        prev_dim = self.target_encoder.fc.weight.shape[1]

        self.target_encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            self.target_encoder.fc,
            nn.BatchNorm1d(dim, affine=False),
        )  # output layer
        # hack: not use bias as it is followed by BN
        self.encoder.fc[6].bias.requires_grad = False

        if init_target_from_online:
            self.target_encoder.load_state_dict(self.encoder.state_dict())

        # disable grad calculations for target model
        set_requires_grad(self.target_encoder, False)

    def forward(self, x1, x2):
        """
        Forward step.

        Parameters
        ----------
        x1 : torch.Tensor
            First view of images.
        x2 : torch.Tensor
            Second view of images.

        Return
        ------
        p1, p2, z1, z2 :
            online predictors and target projections of the networks

        Note
        ----
        See https://arxiv.org/abs/2006.07733 for detailed notations
        """
        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        with torch.no_grad():
            z1_target = self.target_encoder(x1)
            z2_target = self.target_encoder(x2)

        return p1, p2, z1_target, z2_target

    def update_target(self, alpha=None):
        """
        Update target network as the exponential moving average of the encoder.

        Parameters
        ----------
        alpha : float, optional
            The exponential decay weight. The default value is the
            :attr:`alpha` attribute of the object, which was set when it
            was instantiated.
        """
        if alpha is None:
            alpha = self.alpha

        encoder_params = OrderedDict(self.encoder.named_parameters())
        target_params = OrderedDict(self.target_encoder.named_parameters())

        # check if both model contains the same set of keys
        assert encoder_params.keys() == target_params.keys()

        for name, param in target_params.items():
            target_params[name] = ema(param, encoder_params[name], alpha)


# utility functions
def set_requires_grad(model, val):
    for param in model.parameters():
        param.requires_grad = val


def ema(target_param, online_param, alpha):
    if alpha is None:
        return online_param
    return alpha * target_param + (1 - alpha) * online_param
