# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class BYOL(nn.Module):
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
    """

    def __init__(self, base_encoder, init_from_online, dim=2048, pred_dim=512):
        super(BYOL, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        self.target_encoder = base_encoder(num_classes=dim, zero_init_residual=True)

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
        self.encoder.fc[
            6
        ].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build same projector for target model
        self.target_encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False),
        )  # output layer

        self.set_requires_grad(self.target_encoder, False)

        if init_from_online:
            self.update_target(self.target_encoder, self.encoder, None)

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
            online predictors and target projections of the networks

        Note
        ----
        See Sec. 3 of https://arxiv.org/abs/2006.07733 for detailed notations
        """
        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC
        z1_t = self.target_encoder(x1)
        z2_t = self.target_encoder(x2)

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1_t.detach(), z2_t.detach()

    def set_requires_grad(model, val):
        # function taken from:
        # https://github.com/lucidrains/byol-pytorch
        for param in model.parameters():
            param.requires_grad = val

    def update_target(self, target_model, online_model, alpha=0.99):
        def EMA(target_param, online_param, alpha):
            if alpha is None:
                return online_param
            return alpha * target_param + (1 - alpha) * online_param

        target_state_dict = target_model.state_dict()
        for param in target_state_dict:
            target_state_dict[param] = EMA(
                target_state_dict[param], online_model.state_dict()[param], alpha
            )
        target_model.load_state_dict(target_state_dict)
