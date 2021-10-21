import os.path as osp
import random
from glob import glob
from typing import Any, List

import numpy as np
import omegaconf
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class DataModule(LightningDataModule):
    def __init__(self, conf: omegaconf.DictConfig) -> None:
        super().__init__()
        self.conf = conf
        self.max_seq_length = conf.data.max_seq_length

    def prepare_data(self) -> None:
        # download dataset | cache dataset if necessary
        pass

    def gen_dataset(self, tsdata):
        dataset = []
        num_class = [0, 0, 0]
        for idx in range(self.max_seq_length, len(tsdata) - 1):
            sub_tsdata = tsdata[idx - self.max_seq_length : idx]
            assert sub_tsdata.shape[0] == self.max_seq_length

            x = np.asarray([[v] for v in sub_tsdata])
            if sub_tsdata[-1] < tsdata[idx + 1]:  # up
                dataset.append((x, 2))
                num_class[2] += 1
            elif sub_tsdata[-1] > tsdata[idx + 1]:  # down
                dataset.append((x, 0))
                num_class[0] += 1
            else:  # same
                dataset.append((x, 1))
                num_class[1] += 1
        return (dataset, num_class)

    def setup(self) -> None:
        df = pd.read_csv(self.conf.data.data_path)
        df[self.conf.data.column] = df[self.conf.data.column].astype(np.float32)
        dataset, num_class = self.gen_dataset(tsdata=df[self.conf.data.column].values)
        random.shuffle(dataset)

        num_eval = int(len(dataset) * self.conf.data.eval_ratio)
        self.train_dataset = dataset[:-num_eval]
        self.dev_dataset = dataset[-num_eval:]
        return num_class

    def _dataloader(self, dataset: Dataset[Any], shuffle: bool) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            shuffle=shuffle,
            **self.conf.train.dataloader_kwargs,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self.dev_dataset, shuffle=False)
