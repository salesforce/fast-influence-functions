import torch
import numpy as np
import graph_tool as gt
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from graph_tool.draw import graph_draw
from joblib import Parallel, delayed
from transformers import BertTokenizer

from typing import List, Dict, Tuple, Optional
from influence_utils import faiss_utils
from influence_utils import parallel
from experiments.visualization_utils import (
    get_circle_coordinates,
    get_within_circle_constraint,
    # distance_to_points_on_circle,
    # distance_to_points_within_circle,
    distance_to_points_within_circle_vectorized)
from experiments import constants
from experiments import misc_utils
from experiments.hans_utils import HansHelper
from transformers import Trainer, TrainingArguments
from experiments import data_utils
from experiments import mnli_utils

KNN_K = 1000
NUM_EXAMPLES_TO_SHOW = 3


def main(
    train_task_name: str,
    eval_task_name: str,
    num_eval_to_collect: int,
    hans_heuristic: Optional[str] = None,
):

    if train_task_name not in ["mnli-2", "hans"]:
        raise ValueError

    if eval_task_name not in ["mnli-2", "hans"]:
        raise ValueError

    tokenizer, model = misc_utils.create_tokenizer_and_model(
        constants.MNLI2_MODEL_PATH)

    train_dataset, _ = misc_utils.create_datasets(
        task_name=train_task_name,
        tokenizer=tokenizer)

    _, eval_dataset = misc_utils.create_datasets(
        task_name=eval_task_name,
        tokenizer=tokenizer)

    if train_task_name == "mnli-2":
        faiss_index = faiss_utils.FAISSIndex(768, "Flat")
        faiss_index.load(constants.MNLI2_FAISS_INDEX_PATH)
    else:
        faiss_index = None

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp-output",
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            learning_rate=5e-5,
            logging_steps=100),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if eval_task_name in ["mnli-2"]:
        eval_instance_data_loader = misc_utils.get_dataloader(
            dataset=eval_dataset,
            batch_size=1,
            random=False)

    if eval_task_name in ["hans"]:
        if hans_heuristic is None:
            raise ValueError("`hans_heuristic` cannot be None for now")

        hans_helper = HansHelper(
            hans_train_dataset=None,
            hans_eval_dataset=eval_dataset)

        _, eval_instance_data_loader = hans_helper.get_dataset_and_dataloader_of_heuristic(
            mode="eval",
            heuristic=hans_heuristic,
            batch_size=1,
            random=False)

    # Data-points where the model got wrong
    wrong_input_collections = []
    for i, test_inputs in enumerate(eval_instance_data_loader):
        logits, labels, step_eval_loss = misc_utils.predict(
            trainer=trainer,
            model=model,
            inputs=test_inputs)
        if logits.argmax(axis=-1).item() != labels.item():
            wrong_input_collections.append(test_inputs)

    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    if eval_task_name == "mnli-2":
        s_test_damp = 5e-3
        s_test_scale = 1e4
        s_test_num_samples = 1000

    if eval_task_name == "hans":
        s_test_damp = 5e-3
        s_test_scale = 1e6
        s_test_num_samples = 1000

    influences_collections = []
    for index, inputs in enumerate(wrong_input_collections[:num_eval_to_collect]):
        print(f"#{index}")
        if faiss_index is not None:
            features = misc_utils.compute_BERT_CLS_feature(model, **inputs)
            features = features.cpu().detach().numpy()
            KNN_distances, KNN_indices = faiss_index.search(
                k=KNN_K, queries=features)
        else:
            KNN_indices = None

        influences, _ = parallel.compute_influences_parallel(
            # Avoid clash with main process
            device_ids=[0],
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

        influences_collections.append(influences)
        yield influences


def print_most_influential_examples(
        tokenizer: BertTokenizer,
        influences: Dict[int, float],
        train_dataset: data_utils.CustomGlueDataset,
) -> None:
    sorted_indices = misc_utils.sort_dict_keys_by_vals(influences)
    for i in range(NUM_EXAMPLES_TO_SHOW):
        premise, hypothesis, label = mnli_utils.get_inputs_from_features(
            tokenizer=tokenizer,
            label_list=train_dataset.label_list,
            feature=train_dataset[sorted_indices[i]])

        print(f"Most {i}-th influential")
        print(f"\tP:{premise}")
        print(f"\tH:{hypothesis}")
        print(f"\tL:{label}")
