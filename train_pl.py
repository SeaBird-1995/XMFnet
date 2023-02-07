'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-11-18 16:32:06
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import torch
from torch.utils.data import DataLoader
import time
from omegaconf import OmegaConf

from model_pl import XMFNetPL
from dataloader import ViPCDataLoader, collate_fn


def get_git_commit_id():
    """Get the Git commit hash id for logging usage

    Returns:
        str: hash id
    """
    import git
    repo = git.Repo(search_parent_directories=False)
    sha = repo.head.object.hexsha
    return sha


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfg/official.yaml', help='the config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help="the pretrained checkpoint path")
    parser.add_argument('--test_mode', action='store_true', help="whether is a test mode")

    args = parser.parse_args()
    config = OmegaConf.load(args.cfg)
    config.update(vars(args)) # override the configuration using the value in args

    time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    config['Time'] = time_str

    try:
        config['commit_id'] = get_git_commit_id()
    except:
        print("[WARNING] Couldn't get the git commit id")
    
    print(OmegaConf.to_yaml(config, resolve=True))
    return config


config = parse_arguments()

## 1) Define the model
model = XMFNetPL(config)

## 2) Define the dataloader
opt = config.dataset
ViPCDataset_train = ViPCDataLoader(
    'dataset/train_list2.txt', data_path=opt.dataroot, status="train", category=opt.cat)
train_dataloader = DataLoader(ViPCDataset_train,
                              collate_fn=collate_fn,
                              batch_size=opt.batch_size,
                              num_workers=opt.nThreads,
                              shuffle=True,
                              drop_last=True)
ViPCDataset_test = ViPCDataLoader(
    'dataset/test_list2.txt', data_path=opt.dataroot, status="test", category=opt.cat)
val_dataloader = DataLoader(ViPCDataset_test,
                            collate_fn=collate_fn,
                            batch_size=opt.batch_size,
                            num_workers=opt.nThreads,
                            shuffle=True,
                            drop_last=True)

callbacks = []
checkpoint_callback = ModelCheckpoint(
            monitor="val/loss",
            mode='min',
            filename="epoch_{epoch:03d}_val_loss_{val/loss:.5f}",
            auto_insert_metric_name=False,
            save_top_k=1,
            save_last=True,
            verbose=True,
        )
callbacks.append(checkpoint_callback)

trainer = pl.Trainer(devices=4, accelerator="gpu", strategy="ddp", callbacks=callbacks, **config.trainer)
trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=config.checkpoint)

