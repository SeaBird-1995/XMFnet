'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-11-18 12:12:14
Email: haimingzhang@link.cuhk.edu.cn
Description: The Pytorch Lightning model.
'''

import torch
import pytorch_lightning as pl

from model import Network
from decoder.utils.utils import *


def weights_init_normal(m):
    """ Weights initialization with normal distribution.. Xavier """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class XMFNetPL(pl.LightningModule):
    def __init__(self, config):

        super().__init__()

        self.model = Network().apply(weights_init_normal)
        self.loss_cd = L1_ChamferLoss()
        self.loss_cd_eval = L2_ChamferEval_1000()

        self.config = config
        self.save_hyperparameters()
    
    def forward(self, partial, image):
        partial = farthest_point_sample(partial, 2048)
        partial = partial.permute(0, 2, 1)
        complete = self.model(partial, image)
        return complete

    def training_step(self, batch, batch_idx):
        image, gt, partial = batch
        gt = farthest_point_sample(gt, 2048)

        pred_complete = self(partial, image)
        loss_total = self.loss_cd(pred_complete, gt)
        self.log("train/loss", loss_total, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss_total

    def validation_step(self, batch, batch_idx):
        image, gt, partial = batch
        gt = farthest_point_sample(gt, 2048)

        pred_complete = self(partial, image)
        loss_eval = self.loss_cd_eval(pred_complete, gt)
        self.log("val/loss", loss_eval, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optim_opt = self.config.optimizer
        optimizer = torch.optim.Adam(filter(
            lambda p: p.requires_grad, self.model.parameters()), 
            lr=optim_opt.lr, betas=(0.9, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer, milestones=[25, 120], gamma=self.config.lr_scheduler.gamma)
        # return [{
        #         'optimizer': optimizer,
        #         'lr_scheduler': lr_scheduler
        #         }]
        return optimizer
