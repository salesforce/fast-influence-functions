import torch
import pandas as pd
from transformers import (
    BertTokenizer, Trainer, TrainingArguments)
from typing import List, Dict, Tuple, Optional, Callable

from influence_utils import faiss_utils
from influence_utils import parallel
from experiments import constants
from experiments import misc_utils
from experiments import hans_utils
from experiments import data_utils
from experiments import mnli_utils
from influence_utils import nn_influence_utils

KNN_K = 1000
NUM_EXAMPLES_TO_SHOW = 3


class DemoInfluenceHelper(object):
    def __init__(
        self,
        train_task_name: str,
        eval_task_name: str,
        hans_heuristic: Optional[str] = None,
    ) -> None:

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

            hans_helper = hans_utils.HansHelper(
                hans_train_dataset=None,
                hans_eval_dataset=eval_dataset)

            _, eval_instance_data_loader = hans_helper.get_dataset_and_dataloader_of_heuristic(
                mode="eval",
                heuristic=hans_heuristic,
                batch_size=1,
                random=False)

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

        self._model = model
        self._tokenizer = tokenizer
        self._faiss_index = faiss_index
        self._train_dataset = train_dataset
        self._eval_instance_data_loader = eval_instance_data_loader

        self._params_filter = params_filter
        self._weight_decay_ignores = weight_decay_ignores
        self._s_test_damp = s_test_damp
        self._s_test_scale = s_test_scale
        self._s_test_num_samples = s_test_num_samples

    def run(self, chosen_index: int):
        for index, inputs in enumerate(self._eval_instance_data_loader):
            if index != chosen_index:
                break

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.cuda()

        print(f"#{index}")
        if self._faiss_index is not None:
            features = misc_utils.compute_BERT_CLS_feature(
                self._model, **inputs)
            features = features.cpu().detach().numpy()
            KNN_distances, KNN_indices = self._faiss_index.search(
                k=KNN_K, queries=features)
        else:
            KNN_indices = None

        batch_train_data_loader = misc_utils.get_dataloader(
            self._train_dataset,
            batch_size=1,
            random=True)

        instance_train_data_loader = misc_utils.get_dataloader(
            self._train_dataset,
            batch_size=1,
            random=False)

        influences, _, _ = nn_influence_utils.compute_influences(
            n_gpu=1,
            device=torch.device("cuda"),
            batch_train_data_loader=batch_train_data_loader,
            instance_train_data_loader=instance_train_data_loader,
            model=self._model,
            test_inputs=inputs,
            params_filter=self._params_filter,
            weight_decay=constants.WEIGHT_DECAY,
            weight_decay_ignores=self._weight_decay_ignores,
            s_test_damp=self._s_test_damp,
            s_test_scale=self._s_test_scale,
            s_test_num_samples=self._s_test_num_samples,
            train_indices_to_include=KNN_indices,
            precomputed_s_test=None)

        return influences


def print_most_influential_examples(
        tokenizer: BertTokenizer,
        influences: Dict[int, float],
        train_dataset: data_utils.CustomGlueDataset,
        printer_fn: Optional[Callable] = None
) -> None:

    if printer_fn is None:
        printer_fn = print

    sorted_indices = misc_utils.sort_dict_keys_by_vals(influences)
    for i in range(NUM_EXAMPLES_TO_SHOW):
        premise, hypothesis, label = mnli_utils.get_inputs_from_features(
            tokenizer=tokenizer,
            label_list=train_dataset.label_list,
            feature=train_dataset[sorted_indices[i]])

        printer_fn(f"### Most {i}-th influential")
        printer_fn(f"\t**Premise**\n{premise}")
        printer_fn(f"\t**Hypothesis**:{hypothesis}")
        printer_fn(f"\t**Label**:{label}\n")


def load_dataset(name: str) -> pd.DataFrame:
    if name not in ["mnli"]:
        raise ValueError

    if name == "mnli":
        file_name = "/export/share/hguo/Data/Glue/MNLI/dev_matched.tsv"
        columns = ["index", "sentence1", "sentence2", "gold_label"]
    data = pd.read_csv(file_name, sep="\t")
    return data[columns]
