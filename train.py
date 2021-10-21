import os
from typing import List

import omegaconf
from fire import Fire
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from dataloader import DataModule
from model import Model
from utils import read_yaml


def get_callback(conf: omegaconf.DictConfig) -> List[ModelCheckpoint]:
    # TODO: add more callbacks if necessary
    callback_list = list()
    for ckpt_conf in conf.callback.model_checkpoint:
        callback_list.append(ModelCheckpoint(**ckpt_conf))
    return callback_list


def train(conf_path: str = "conf/conf.yaml", gpus: str = "2", seed: int = 42) -> None:
    seed_everything(seed)
    # load conf
    conf = OmegaConf.create(read_yaml(conf_path))
    conf.train.trainer.gpus = gpus

    # load datamodule

    datamodule = DataModule(conf=conf)
    num_class = datamodule.setup()

    # load model
    model = Model(conf, num_class)
    #
    # load callbacks
    callbacks = get_callback(conf)
    # Initialize Trainer
    trainer = Trainer(
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=False),
        **conf.train.trainer
    )
    # Train
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    Fire(train)
