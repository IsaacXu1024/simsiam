# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
    Build a BYOL model.

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
    """

    def __init__(
        self, base_encoder, dim=2048, pred_dim=512, init_target_from_online=False
    ):
        super(BYOL, self).__init__(base_encoder, dim, pred_dim)

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

    def update_target(self, alpha=0.99):
        target_state_dict = self.target_encoder.state_dict()
        for param in target_state_dict:
            target_state_dict[param] = ema(
                target_state_dict[param], self.encoder.state_dict()[param], alpha
            )
        self.target_encoder.load_state_dict(target_state_dict)


# utility functions
def set_requires_grad(model, val):
    for param in model.parameters():
        param.requires_grad = val


def ema(target_param, online_param, alpha):
    if alpha is None:
        return online_param
    return alpha * target_param + (1 - alpha) * online_param
