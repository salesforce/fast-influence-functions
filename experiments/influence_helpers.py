import torch
from influence_utils import faiss_utils
from typing import List, Dict, Tuple, Optional, Union, Any

from experiments import constants
from experiments import misc_utils
from influence_utils import parallel
from influence_utils import nn_influence_utils


def compute_influences_simplified(
        k: int,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        trained_on_task_name: str,
        eval_task_name: str,
        train_dataset: torch.utils.data.DataLoader,
        use_parallel: bool,
        device_ids: Optional[List[int]] = None,
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
) -> Tuple[Dict[int, float], List[torch.FloatTensor]]:

    faiss_index = faiss_utils.FAISSIndex(768, "Flat")
    faiss_index.load(constants.MNLI_FAISS_INDEX_PATH)
    print(f"Loaded FAISS index with {len(faiss_index)} entries")

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

    # Other settings are not supported as of now
    if trained_on_task_name == "mnli-2" and eval_task_name == "mnli-2":
        s_test_damp = 5e-3
        s_test_scale = 1e4
        s_test_num_samples = 1000

    if trained_on_task_name == "hans" and eval_task_name == "hans":
        s_test_damp = 5e-3
        s_test_scale = 1e6
        s_test_num_samples = 2000

    if trained_on_task_name == "mnli-2" and eval_task_name == "hans":
        s_test_damp = 5e-3
        s_test_scale = 1e6
        s_test_num_samples = 1000

    if trained_on_task_name == "hans" and eval_task_name == "mnli-2":
        s_test_damp = 5e-3
        s_test_scale = 1e6
        s_test_num_samples = 2000

    if faiss_index is not None:
        features = misc_utils.compute_BERT_CLS_feature(model, **inputs)
        features = features.cpu().detach().numpy()
        KNN_distances, KNN_indices = faiss_index.search(
            k=k, queries=features)
    else:
        KNN_indices = None

    if not use_parallel:
        model.cuda()
        batch_train_data_loader = misc_utils.get_dataloader(
            train_dataset,
            batch_size=1,
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
            train_indices_to_include=KNN_indices,
            precomputed_s_test=None)
    else:
        influences, _ = parallel.compute_influences_parallel(
            # Avoid clash with main process
            device_ids=[0, 1, 2, 3],
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
