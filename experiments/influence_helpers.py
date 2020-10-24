import torch
from influence_utils import faiss_utils
from typing import List, Dict, Tuple, Optional, Union, Any

from experiments import constants
from experiments import misc_utils
from influence_utils import nn_influence_utils


def compute_influences_simplified(
        k: int,
        model: torch.nn.Module,
        test_inputs: Dict[str, torch.Tensor],
        batch_train_data_loader: torch.utils.data.DataLoader,
        instance_train_data_loader: torch.utils.data.DataLoader,
        device_ids: Optional[List[int]] = None,
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
) -> Tuple[Dict[int, float], List[torch.FloatTensor]]:

    faiss_index = faiss_utils.FAISSIndex(768, "Flat")
    faiss_index.load(constants.MNLI_FAISS_INDEX_PATH)
    print(f"Loaded FAISS index with {len(faiss_index)} entries")

    test_features = misc_utils.compute_BERT_CLS_feature(model, **test_inputs)
    test_features = test_features.cpu().detach().numpy()
    KNN_distances, KNN_indices = faiss_index.search(
        k=k, queries=test_features)

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

    if device_ids is None:
        (influences,
         train_inputs_collections,
         s_test) = nn_influence_utils.compute_influences(
            n_gpu=1,
            device=torch.device("cuda"),
            model=model,
            test_inputs=test_inputs,
            batch_train_data_loader=batch_train_data_loader,
            instance_train_data_loader=instance_train_data_loader,
            params_filter=params_filter,
            weight_decay=constants.WEIGHT_DECAY,
            weight_decay_ignores=weight_decay_ignores,
            s_test_scale=1000,
            s_test_num_samples=300,
            precomputed_s_test=precomputed_s_test,
            train_indices_to_include=KNN_indices)
    else:
        raise ValueError("Deprecated")

    return influences, s_test
