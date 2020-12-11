# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch
import numpy as np
import pandas as pd

from experiments import constants
from experiments import data_utils
from experiments import misc_utils
from transformers import default_data_collator
from typing import List, Union, Iterable, Dict, Any, Tuple, Optional


class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset: data_utils.CustomGlueDataset,
                 indices: Union[np.ndarray, List[int]]) -> None:

        super(SubsetDataset, self).__init__()
        self.wrapped_dataset = dataset
        self.indices = indices

    def __getitem__(self, index) -> Dict[str, Union[torch.Tensor, Any]]:
        mapped_index = self.indices[index]
        return self.wrapped_dataset[mapped_index]

    def __len__(self) -> int:
        return len(self.indices)


class HansHelper(object):
    def __init__(
            self,
            hans_train_dataset: Optional[data_utils.CustomGlueDataset] = None,
            hans_eval_dataset: Optional[data_utils.CustomGlueDataset] = None) -> None:

        self._hans_train_df = pd.read_csv(constants.HANS_TRAIN_FILE_NAME, sep="\t")
        self._hans_eval_df = pd.read_csv(constants.HANS_EVAL_FILE_NAME, sep="\t")
        self._hans_train_dataset = hans_train_dataset
        self._hans_eval_dataset = hans_eval_dataset

    def get_indices_of_heuristic(
            self,
            mode: str,
            heuristic: str) -> List[int]:

        if mode not in ["train", "eval"]:
            raise ValueError

        if heuristic not in ["lexical_overlap", "subsequence", "constituent"]:
            raise ValueError

        if mode == "train":
            df = self._hans_train_df
        else:
            df = self._hans_eval_df

        indices_of_heuristic = df[df.heuristic == heuristic].index
        return indices_of_heuristic.tolist()

    def sample_batch_of_heuristic(
            self,
            mode: str,
            heuristic: str,
            size: int,
            return_raw_data: bool = False) -> np.ndarray:

        if mode not in ["train", "eval"]:
            raise ValueError

        if mode == "train":
            dataset = self._hans_train_dataset
        else:
            dataset = self._hans_eval_dataset

        if dataset is None:
            raise ValueError("`dataset` is None")

        indices = self.get_indices_of_heuristic(
            mode=mode, heuristic=heuristic)

        sampled_indices = np.random.choice(
            indices, size=size, replace=False)

        sampled_data = [dataset[index] for index in sampled_indices]
        batched_data = default_data_collator(sampled_data)
        if return_raw_data is False:
            return batched_data

        return batched_data, sampled_data

    def get_dataset_and_dataloader_of_heuristic(
            self,
            mode: str,
            heuristic: str,
            batch_size: int,
            random: bool) -> Tuple[SubsetDataset,
                                   torch.utils.data.DataLoader]:

        if mode not in ["train", "eval"]:
            raise ValueError

        if mode == "train":
            dataset = self._hans_train_dataset
        else:
            dataset = self._hans_eval_dataset

        if dataset is None:
            raise ValueError("`dataset` is None")

        indices = self.get_indices_of_heuristic(
            mode=mode, heuristic=heuristic)

        heuristic_dataset = SubsetDataset(dataset=dataset, indices=indices)
        heuristic_dataloader = misc_utils.get_dataloader(
            dataset=heuristic_dataset,
            batch_size=batch_size,
            random=random)

        return heuristic_dataset, heuristic_dataloader
