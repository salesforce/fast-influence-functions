# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch
import numpy as np
import pandas as pd
from collections import defaultdict

from experiments import constants
from experiments import data_utils
from experiments import misc_utils
from transformers import default_data_collator
from typing import List, Union, Iterable, Dict, Any, Tuple, Optional

try:
    from wilds.datasets.amazon_dataset import AmazonDataset
except ModuleNotFoundError:
    AmazonDataset = None


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

        # This file includes both validation and test
        combined_hans_eval_df = pd.read_csv(constants.HANS_EVAL_FILE_NAME, sep="\t")
        # This is a list of indices that should be mapped to validation dataset
        valid_indices = torch.load(constants.HANS_VALID_INDICES_FILE_NAME)
        # https://stackoverflow.com/questions/28256761/select-pandas-rows-by-excluding-index-number
        valid_selector = combined_hans_eval_df.index.isin(valid_indices)

        self._hans_train_df = pd.read_csv(constants.HANS_TRAIN_FILE_NAME, sep="\t")
        self._hans_eval_df = combined_hans_eval_df[valid_selector]
        self._hans_test_df = combined_hans_eval_df[~valid_selector]
        self._hans_train_dataset = hans_train_dataset
        self._hans_eval_dataset = hans_eval_dataset

    def get_indices_of_heuristic(
            self,
            mode: str,
            heuristic: str) -> List[int]:

        if mode not in ["train", "eval", "test"]:
            raise ValueError

        if heuristic not in ["lexical_overlap", "subsequence", "constituent"]:
            raise ValueError

        if mode == "train":
            df = self._hans_train_df
        if mode == "eval":
            df = self._hans_eval_df
        if mode == "test":
            df = self._hans_test_df

        indices_of_heuristic = df[df.heuristic == heuristic].index
        return indices_of_heuristic.tolist()

    def sample_batch_of_heuristic(
            self,
            mode: str,
            heuristic: str,
            size: int,
            return_raw_data: bool = False) -> np.ndarray:

        if mode not in ["train", "eval", "test"]:
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

        if mode not in ["train", "eval", "test"]:
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


class SimpleHelper(object):
    def __init__(
            self,
            train_dataset: Optional[data_utils.CustomGlueDataset] = None,
            eval_dataset: Optional[data_utils.CustomGlueDataset] = None,
            test_dataset: Optional[data_utils.CustomGlueDataset] = None) -> None:

        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._test_dataset = test_dataset

    def sample_batch_of_heuristic(
            self,
            mode: str,
            heuristic: str,
            size: int,
            return_raw_data: bool = False) -> np.ndarray:

        if mode not in ["train", "eval", "test"]:
            raise ValueError

        if heuristic not in ["null"]:
            raise ValueError

        if mode == "train":
            dataset = self._train_dataset
        if mode == "eval":
            dataset = self._eval_dataset
        if mode == "test":
            dataset = self._test_dataset

        if dataset is None:
            raise ValueError("`dataset` is None")

        sampled_indices = np.random.choice(
            len(dataset), size=size, replace=False)

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

        if mode not in ["train", "eval", "test"]:
            raise ValueError

        if heuristic not in ["null"]:
            raise ValueError

        if mode == "train":
            dataset = self._train_dataset
        if mode == "eval":
            dataset = self._eval_dataset
        if mode == "test":
            dataset = self._test_dataset

        if dataset is None:
            raise ValueError("`dataset` is None")

        heuristic_dataloader = misc_utils.get_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            random=random)

        return dataset, heuristic_dataloader


class AmazonHelper(SimpleHelper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # This is a dictionary that maps split-key
        # into the corresponding metadata array.
        metadata_arrays_dict = torch.load(
            constants.AMAZON_METADATA_ARRAY_FILE_NAME)
        # The first column refers to `user`
        valid_user_ids = metadata_arrays_dict["val"][:, 0].numpy()
        # Create a mapping from `user_id` to the
        # data-indices of this `user_id`
        valid_user_to_index_map = defaultdict(list)
        for index, user_id in enumerate(valid_user_ids):
            valid_user_to_index_map[user_id].append(index)

        self._valid_user_to_index_map = valid_user_to_index_map

    def generate_sampled_indices(self, mode: str, size: int) -> List[int]:
        if mode not in ["eval"]:
            raise ValueError

        # For now, we only sample one index per `user_id`
        if size > len(self._valid_user_to_index_map.keys()):
            raise ValueError

        sampled_indices = []
        # First sample the `user_id`
        sampled_user_ids = np.random.choice(
            list(self._valid_user_to_index_map.keys()),
            size=size,
            replace=False)

        # Then sample data indices from the sampled `user_id`
        for sampled_user_id in sampled_user_ids:
            _sampled_indices = np.random.choice(
                self._valid_user_to_index_map[sampled_user_id],
                size=1,
                replace=False)
            sampled_indices.extend(_sampled_indices)

        return sampled_indices

    def sample_batch_of_heuristic(
            self,
            mode: str,
            heuristic: str,
            size: int,
            return_raw_data: bool = False) -> np.ndarray:

        if mode not in ["train", "eval", "test"]:
            raise ValueError

        if heuristic not in ["null"]:
            raise ValueError

        if mode == "train":
            dataset = self._train_dataset
        if mode == "eval":
            dataset = self._eval_dataset
        if mode == "test":
            dataset = self._test_dataset

        if dataset is None:
            raise ValueError("`dataset` is None")

        sampled_indices = self.generate_sampled_indices(mode=mode, size=size)
        sampled_data = [dataset[index] for index in sampled_indices]
        batched_data = default_data_collator(sampled_data)
        if return_raw_data is False:
            return batched_data

        return batched_data, sampled_data


def save_amazon_metadata(file_name: str) -> None:
    dataset = AmazonDataset(
        root_dir=constants.Amazon_DATA_DIR,
        download=False)

    metadata_arrays_dict = {}
    for key in dataset.split_dict.keys():
        datasubset = dataset.get_subset(key)
        metadata_arrays_dict[key] = datasubset.metadata_array
        print(f"{key:<10}: {len(datasubset):<10} "
              f"{datasubset.metadata_array.shape} "
              f"{metadata_arrays_dict[key].shape}")

    # print(torch.__version__)
    torch.save(metadata_arrays_dict, file_name)
