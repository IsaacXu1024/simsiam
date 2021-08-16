# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import builder
import torch
import torch.nn as nn
import utilities


class BYOL(builder.SimSiam):
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

    def __init__(self, base_encoder, init_target_from_online, dim=2048, pred_dim=512):
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

        if init_target_from_online:
            self.target_encoder.load_state_dict(self.encoder.state_dict())

        # disable grad calculations for target model
        utilities.set_requires_grad(self.target_encoder, False)

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
        z1_target = self.target_encoder(x1)
        z2_target = self.target_encoder(x2)

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1_target, z2_target

    def update_target(self, target_model, online_model, alpha=0.99):
        target_state_dict = target_model.state_dict()
        for param in target_state_dict:
            target_state_dict[param] = utilities.ema(
                target_state_dict[param], online_model.state_dict()[param], alpha
            )
        target_model.load_state_dict(target_state_dict)
