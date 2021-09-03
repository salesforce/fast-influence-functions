# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch
from influence_utils import faiss_utils
from typing import List, Dict, Tuple, Optional, Union, Any

from experiments import constants
from experiments import misc_utils
from influence_utils import parallel
from influence_utils import nn_influence_utils


def load_faiss_index(
        trained_on_task_name: str,
        train_task_name: str,
) -> faiss_utils.FAISSIndex:

    if trained_on_task_name not in ["mnli", "mnli-2", "hans", "amazon"]:
        raise ValueError

    if train_task_name not in ["mnli", "mnli-2", "hans", "amazon", "anli"]:
        raise ValueError

    if trained_on_task_name == "mnli" and train_task_name == "mnli":
        faiss_index = faiss_utils.FAISSIndex(768, "Flat")
        faiss_index.load(constants.MNLI_FAISS_INDEX_PATH)

    elif trained_on_task_name == "mnli-2" and train_task_name == "mnli-2":
        faiss_index = faiss_utils.FAISSIndex(768, "Flat")
        faiss_index.load(constants.MNLI2_FAISS_INDEX_PATH)

    elif trained_on_task_name == "hans" and train_task_name == "hans":
        faiss_index = faiss_utils.FAISSIndex(768, "Flat")
        faiss_index.load(constants.HANS_FAISS_INDEX_PATH)

    elif trained_on_task_name == "mnli-2" and train_task_name == "hans":
        faiss_index = faiss_utils.FAISSIndex(768, "Flat")
        faiss_index.load(constants.MNLI2_HANS_FAISS_INDEX_PATH)

    elif trained_on_task_name == "hans" and train_task_name == "mnli-2":
        faiss_index = faiss_utils.FAISSIndex(768, "Flat")
        faiss_index.load(constants.HANS_MNLI2_FAISS_INDEX_PATH)

    elif trained_on_task_name == "amazon" and train_task_name == "amazon":
        faiss_index = faiss_utils.FAISSIndex(768, "Flat")
        faiss_index.load(constants.Amazon_FAISS_INDEX_PATH)

    elif trained_on_task_name == "mnli" and train_task_name == "anli":
        faiss_index = faiss_utils.FAISSIndex(768, "Flat")
        faiss_index.load(constants.MNLI_ANLI_FAISS_INDEX_PATH)

    else:
        faiss_index = None

    return faiss_index


def select_s_test_config(
        trained_on_task_name: str,
        train_task_name: str,
        eval_task_name: str,
) -> Tuple[float, float, int]:

    if trained_on_task_name != train_task_name:
        # Only this setting is supported for now
        # basically, the config for this combination
        # of `trained_on_task_name` and `eval_task_name`
        # would be fine, so not raising issues here for now.
        if not (
            all([trained_on_task_name == "mnli-2",
                 train_task_name == "hans",
                 eval_task_name == "hans"]) or
            all([trained_on_task_name == "mnli",
                 train_task_name == "anli",
                 eval_task_name == "anli"])
        ):
            raise ValueError("Unsupported as of now")

    if trained_on_task_name not in ["mnli", "mnli-2", "hans", "amazon"]:
        raise ValueError

    if eval_task_name not in ["mnli", "mnli-2", "hans", "amazon", "anli"]:
        raise ValueError

    # Other settings are not supported as of now
    if trained_on_task_name == "mnli" and eval_task_name == "mnli":
        s_test_damp = 5e-3
        s_test_scale = 1e4
        s_test_num_samples = 1000

    elif trained_on_task_name == "mnli-2" and eval_task_name == "mnli-2":
        s_test_damp = 5e-3
        s_test_scale = 1e4
        s_test_num_samples = 1000

    elif trained_on_task_name == "hans" and eval_task_name == "hans":
        s_test_damp = 5e-3
        s_test_scale = 1e6
        s_test_num_samples = 2000

    elif trained_on_task_name == "mnli-2" and eval_task_name == "hans":
        s_test_damp = 5e-3
        s_test_scale = 1e6
        s_test_num_samples = 1000

    elif trained_on_task_name == "hans" and eval_task_name == "mnli-2":
        s_test_damp = 5e-3
        s_test_scale = 1e6
        s_test_num_samples = 2000

    elif trained_on_task_name == "amazon" and eval_task_name == "amazon":
        s_test_damp = 5e-3
        s_test_scale = 1e4
        s_test_num_samples = 1000

    elif trained_on_task_name == "mnli" and eval_task_name == "anli":
        if train_task_name != "anli":
            raise NotImplementedError

        s_test_damp = 5e-3
        s_test_scale = 1e6
        s_test_num_samples = 1500

    else:
        raise ValueError

    return s_test_damp, s_test_scale, s_test_num_samples


def compute_influences_simplified(
        k: int,
        faiss_index: Optional[faiss_utils.FAISSIndex],
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        train_dataset: torch.utils.data.DataLoader,
        use_parallel: bool,
        s_test_damp: float,
        s_test_scale: float,
        s_test_num_samples: int,
        device_ids: Optional[List[int]] = None,
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
        faiss_index_use_mean_features_as_query: bool = False,
        _batch_size: int = 1,
        _s_test_iterations: int = 1,
) -> Dict[int, float]:

    # Make sure indices are sorted according to distances
    # KNN_distances[(
    #     KNN_indices.squeeze(axis=0)[
    #         np.argsort(KNN_distances.squeeze(axis=0))
    #     ] != KNN_indices)]

    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    if faiss_index is not None:
        features = misc_utils.compute_BERT_CLS_feature(model, **inputs)
        features = features.cpu().detach().numpy()

        if faiss_index_use_mean_features_as_query is True:
            # We use the mean embedding as the final query here
            features = features.mean(axis=0, keepdims=True)

        KNN_distances, KNN_indices = faiss_index.search(
            k=k, queries=features)
    else:
        KNN_indices = None

    if not use_parallel:
        model.cuda()
        batch_train_data_loader = misc_utils.get_dataloader(
            train_dataset,
            batch_size=_batch_size,
            random=True)

        instance_train_data_loader = misc_utils.get_dataloader(
            train_dataset,
            batch_size=1,
            random=False)

        influences, _, _ = nn_influence_utils.compute_influences(
            n_gpu=1,
            device=torch.device("cuda"),
            batch_train_data_loader=batch_train_data_loader,
            instance_train_data_loader=instance_train_data_loader,
            model=model,
            test_inputs=inputs,
            params_filter=params_filter,
            weight_decay=constants.WEIGHT_DECAY,
            weight_decay_ignores=weight_decay_ignores,
            s_test_damp=s_test_damp,
            s_test_scale=s_test_scale,
            s_test_num_samples=s_test_num_samples,
            s_test_iterations=_s_test_iterations,
            train_indices_to_include=KNN_indices,
            precomputed_s_test=precomputed_s_test)
    else:
        if device_ids is None:
            raise ValueError("`device_ids` cannot be None")

        influences, _ = parallel.compute_influences_parallel(
            # Avoid clash with main process
            device_ids=device_ids,
            train_dataset=train_dataset,
            batch_size=1,
            model=model,
            test_inputs=inputs,
            params_filter=params_filter,
            weight_decay=constants.WEIGHT_DECAY,
            weight_decay_ignores=weight_decay_ignores,
            s_test_damp=s_test_damp,
            s_test_scale=s_test_scale,
            s_test_num_samples=s_test_num_samples,
            train_indices_to_include=KNN_indices,
            return_s_test=False,
            debug=False)

    return influences
