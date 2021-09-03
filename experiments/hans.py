# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import torch
import numpy as np
import transformers
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from transformers import InputFeatures
from transformers import default_data_collator
from typing import Union, Dict, Any, List, Tuple, Optional

from influence_utils import faiss_utils
from influence_utils import nn_influence_utils
from experiments import constants
from experiments import misc_utils
# from experiments import remote_utils
from experiments import influence_helpers
from experiments.hans_utils import HansHelper, SimpleHelper, AmazonHelper
from transformers import TrainingArguments
from experiments.data_utils import CustomGlueDataset

DEFAULT_KNN_K = 1000
DEFAULT_NUM_REPLICAS = 3
EVAL_HEURISTICS_SAMPLE_BATCH_SIZE = 10
EXPERIMENT_TYPES = ["most-helpful", "most-harmful", "random"]
DEFAULT_HANS_EVAL_HEURISTICS = ["lexical_overlap", "subsequence", "constituent"]
DEFAULT_ANLI_EVAL_HEURISTICS = ["null"]
DEFAULT_Amazon_EVAL_HEURISTICS = ["null"]
VERSION_2_NUM_DATAPOINTS_CHOICES = [EVAL_HEURISTICS_SAMPLE_BATCH_SIZE]
VERSION_2_LEARNING_RATE_CHOICES = [1e-4]


def main(
        trained_on_task_name: str,
        train_task_name: str,
        train_heuristic: str,
        num_replicas: Optional[int] = None,
        use_parallel: bool = True,
        version: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:

    if trained_on_task_name not in ["mnli", "mnli-2", "amazon"]:
        raise ValueError

    if train_task_name not in ["mnli-2", "hans", "amazon", "anli"]:
        raise ValueError

    if num_replicas is None:
        num_replicas = DEFAULT_NUM_REPLICAS

    if version not in ["new-only-z", "new-only-ztest", "new-z-and-ztest"]:
        raise ValueError

    if trained_on_task_name in ["mnli-2"]:
        eval_heuristics = DEFAULT_HANS_EVAL_HEURISTICS
        task_tokenizer, task_model = misc_utils.create_tokenizer_and_model(
            constants.MNLI2_MODEL_PATH)

        (mnli_train_dataset,
         mnli_eval_dataset) = misc_utils.create_datasets(
            task_name="mnli-2",
            tokenizer=task_tokenizer)

        (hans_train_dataset,
         hans_eval_dataset) = misc_utils.create_datasets(
            task_name="hans",
            tokenizer=task_tokenizer)

        if train_task_name == "mnli-2":
            train_dataset = mnli_train_dataset

        if train_task_name == "hans":
            train_dataset = hans_train_dataset

        (s_test_damp,
         s_test_scale,
         s_test_num_samples) = influence_helpers.select_s_test_config(
            trained_on_task_name=trained_on_task_name,
            train_task_name=train_task_name,
            eval_task_name="hans",
        )

        hans_helper = HansHelper(
            hans_train_dataset=hans_train_dataset,
            hans_eval_dataset=hans_eval_dataset)

    if trained_on_task_name in ["amazon"]:
        # This is not used, so used `null` as a placeholder
        eval_heuristics = DEFAULT_Amazon_EVAL_HEURISTICS
        task_tokenizer, task_model = misc_utils.create_tokenizer_and_model(
            constants.Amazon_MODEL_PATH)

        (amazon_train_dataset,
         amazon_eval_dataset,
         amazon_test_dataset) = misc_utils.create_datasets(
            task_name="amazon",
            tokenizer=task_tokenizer,
            create_test_dataset=True)

        # Fine-tune on training dataset
        train_dataset = amazon_train_dataset

        (s_test_damp,
         s_test_scale,
         s_test_num_samples) = influence_helpers.select_s_test_config(
            trained_on_task_name=trained_on_task_name,
            train_task_name=train_task_name,
            eval_task_name="amazon",
        )

        hans_helper = AmazonHelper(
            train_dataset=amazon_train_dataset,
            eval_dataset=amazon_eval_dataset,
            test_dataset=amazon_test_dataset)

    if trained_on_task_name in ["mnli"]:
        # This is not used, so used `null` as a placeholder
        eval_heuristics = DEFAULT_ANLI_EVAL_HEURISTICS
        task_tokenizer, task_model = misc_utils.create_tokenizer_and_model(
            constants.MNLI_MODEL_PATH)

        (anli_train_dataset,
         anli_eval_dataset,
         anli_test_dataset) = misc_utils.create_datasets(
            task_name="anli",
            tokenizer=task_tokenizer,
            create_test_dataset=True)

        # Fine-tune on training dataset
        train_dataset = anli_train_dataset

        (s_test_damp,
         s_test_scale,
         s_test_num_samples) = influence_helpers.select_s_test_config(
            trained_on_task_name=trained_on_task_name,
            train_task_name=train_task_name,
            eval_task_name="anli",
        )

        hans_helper = SimpleHelper(
            train_dataset=anli_train_dataset,
            eval_dataset=anli_eval_dataset,
            test_dataset=anli_test_dataset)

    # We will be running model trained on MNLI-2
    # but calculate influences on HANS dataset
    faiss_index = influence_helpers.load_faiss_index(
        trained_on_task_name=trained_on_task_name,
        train_task_name=train_task_name
    )

    # Most of these arguments are placeholders
    # and are not really used at all, so ignore
    # the exact values of these.
    trainer = transformers.Trainer(
        model=task_model,
        args=TrainingArguments(
            output_dir="./tmp-output",
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            learning_rate=5e-5,
            logging_steps=100),
    )

    output_collections: Dict[str, List] = defaultdict(list)

    if version == "old":
        raise ValueError("Deprecated")

    else:
        NUM_STEPS = 10
        num_total_experiments = (
            len(EXPERIMENT_TYPES) *
            num_replicas *
            len(VERSION_2_NUM_DATAPOINTS_CHOICES) *
            len(VERSION_2_LEARNING_RATE_CHOICES) *
            NUM_STEPS
        )

        with tqdm(total=num_total_experiments) as pbar:
            for experiment_type in EXPERIMENT_TYPES:
                for replica_index in range(num_replicas):
                    for version_2_num_datapoints in VERSION_2_NUM_DATAPOINTS_CHOICES:
                        for version_2_learning_rate in VERSION_2_LEARNING_RATE_CHOICES:

                            # The model will be used for multiple
                            # steps so `deepcopy` it here.
                            _model = deepcopy(task_model)
                            for step in range(NUM_STEPS):

                                # Sample anchor data-points every step
                                (hans_eval_heuristic_inputs,
                                 hans_eval_heuristic_raw_inputs) = hans_helper.sample_batch_of_heuristic(
                                    mode="eval",
                                    heuristic=train_heuristic,
                                    size=EVAL_HEURISTICS_SAMPLE_BATCH_SIZE,
                                    return_raw_data=True)

                                misc_utils.move_inputs_to_device(
                                    inputs=hans_eval_heuristic_inputs,
                                    device=task_model.device)

                                outputs_one_experiment, _model = one_experiment(
                                    use_parallel=use_parallel,
                                    eval_heuristics=eval_heuristics,
                                    experiment_type=experiment_type,
                                    hans_helper=hans_helper,
                                    train_dataset=train_dataset,
                                    task_model=_model,
                                    faiss_index=faiss_index,
                                    s_test_damp=s_test_damp,
                                    s_test_scale=s_test_scale,
                                    s_test_num_samples=s_test_num_samples,
                                    trainer=trainer,
                                    version=version,
                                    version_2_num_datapoints=version_2_num_datapoints,
                                    version_2_learning_rate=version_2_learning_rate,
                                    hans_eval_heuristic_inputs=hans_eval_heuristic_inputs,
                                    hans_eval_heuristic_raw_inputs=hans_eval_heuristic_raw_inputs,
                                )

                                output_collections[
                                    f"{experiment_type}-"
                                    f"{replica_index}-"
                                    f"{version_2_num_datapoints}-"
                                    f"{version_2_learning_rate}-"
                                ].append(outputs_one_experiment)

                                pbar.update(1)
                                pbar.set_description(f"{experiment_type} #{replica_index}")

        torch.save(
            output_collections,
            f"hans-augmentation-{version}."
            f"{trained_on_task_name}."
            f"{train_task_name}."
            f"{train_heuristic}."
            f"{num_replicas}."
            f"{use_parallel}."
            f"{DEFAULT_KNN_K}.pth")

    return output_collections


def one_experiment(
    use_parallel: bool,
    eval_heuristics: List[str],
    experiment_type: str,
    hans_helper: HansHelper,
    train_dataset: CustomGlueDataset,
    task_model: torch.nn.Module,
    faiss_index: faiss_utils.FAISSIndex,
    s_test_damp: float,
    s_test_scale: float,
    s_test_num_samples: int,
    trainer: transformers.Trainer,
    version: str,
    version_2_num_datapoints: Optional[int],
    version_2_learning_rate: Optional[float],
    hans_eval_heuristic_inputs: Dict[str, Any],
    hans_eval_heuristic_raw_inputs: List[InputFeatures],
) -> Tuple[Dict[str, Any], Optional[torch.nn.Module]]:
    if task_model.device.type != "cuda":
        raise ValueError("The model is supposed to be on CUDA")

    if version_2_num_datapoints is None:
        raise ValueError
    if version_2_learning_rate is None:
        raise ValueError

    if experiment_type in ["most-harmful", "most-helpful"]:

        influences = influence_helpers.compute_influences_simplified(
            k=DEFAULT_KNN_K,
            faiss_index=faiss_index,
            model=task_model,
            inputs=hans_eval_heuristic_inputs,
            train_dataset=train_dataset,
            use_parallel=use_parallel,
            s_test_damp=s_test_damp,
            s_test_scale=s_test_scale,
            s_test_num_samples=s_test_num_samples,
            device_ids=[1, 2, 3],
            precomputed_s_test=None,
            faiss_index_use_mean_features_as_query=True,
        )
        helpful_indices, harmful_indices = misc_utils.get_helpful_harmful_indices_from_influences_dict(
            influences, n=version_2_num_datapoints)
        if experiment_type == "most-helpful":
            datapoint_indices = helpful_indices

        if experiment_type == "most-harmful":
            datapoint_indices = harmful_indices

    if experiment_type == "random":
        # s_test = None
        influences = None
        hans_eval_heuristic_inputs = None
        # Essentially shuffle the indices
        datapoint_indices = np.random.choice(
            len(train_dataset),
            size=len(train_dataset),
            replace=False)

    loss_collections = {}
    accuracy_collections = {}

    # num_datapoints = 1
    # learning_rate = 1e-4
    num_datapoints = version_2_num_datapoints
    learning_rate = version_2_learning_rate

    if version == "new-only-z":
        datapoints = [
            train_dataset[index]
            for index in datapoint_indices[:num_datapoints]]

    if version == "new-only-ztest":
        datapoints = hans_eval_heuristic_raw_inputs

    if version == "new-z-and-ztest":
        datapoints = [
            train_dataset[index]
            for index in datapoint_indices[:num_datapoints]
        ] + hans_eval_heuristic_raw_inputs

    batch = default_data_collator(datapoints)
    new_model, _ = pseudo_gradient_step(
        model=task_model,
        inputs=batch,
        learning_rate=learning_rate)

    for heuristic in eval_heuristics:
        new_model_loss, new_model_accuracy = evaluate_heuristic(
            hans_helper=hans_helper,
            heuristic=heuristic,
            trainer=trainer,
            model=new_model)

        loss_collections[heuristic] = new_model_loss
        accuracy_collections[heuristic] = new_model_accuracy
        # print(f"Finished {num_datapoints}-{learning_rate}")

    output_collections = {
        # "s_test": s_test,
        "influences": influences,
        "loss": loss_collections,
        "accuracy": accuracy_collections,
        "datapoint_indices": datapoint_indices,
        "learning_rate": learning_rate,
        "num_datapoints": num_datapoints,
        "hans_eval_heuristic_inputs": hans_eval_heuristic_inputs,
    }

    # Warning: Check again whether using this `new_model` is a good idea
    return output_collections, new_model


def pseudo_gradient_step(
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        learning_rate: float,
        precomputed_gradients_z: Optional[List[torch.FloatTensor]] = None
) -> Tuple[torch.nn.Module, List[torch.FloatTensor]]:

    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    params_to_freeze = [
        "bert.embeddings.",
        "bert.encoder.layer.0.",
        "bert.encoder.layer.1.",
        "bert.encoder.layer.2.",
        "bert.encoder.layer.3.",
        "bert.encoder.layer.4.",
        "bert.encoder.layer.5.",
        "bert.encoder.layer.6.",
        "bert.encoder.layer.7.",
        "bert.encoder.layer.8.",
        "bert.encoder.layer.9.",
    ]

    if precomputed_gradients_z is not None:
        gradients_z = precomputed_gradients_z
    else:
        gradients_z = nn_influence_utils.compute_gradients(
            n_gpu=1,
            device=torch.device("cuda"),
            model=model,
            inputs=inputs,
            params_filter=params_filter,
            weight_decay=constants.WEIGHT_DECAY,
            weight_decay_ignores=weight_decay_ignores)

    new_model = deepcopy(model)
    params_to_update = [
        p for name, p in new_model.named_parameters()
        if not any(pfreeze in name for pfreeze in params_to_freeze)]

    # They should refer to the same parameters
    if len(params_to_update) != len(gradients_z):
        raise ValueError

    with torch.no_grad():
        [p.sub_(learning_rate * grad_z) for p, grad_z in
         zip(params_to_update, gradients_z)]

    return new_model, gradients_z


def evaluate_heuristic(
        hans_helper: HansHelper,
        heuristic: str,
        trainer: transformers.Trainer,
        model: torch.nn.Module,
) -> Tuple[float, float]:

    _, batch_dataloader = hans_helper.get_dataset_and_dataloader_of_heuristic(
        mode="test",
        heuristic=heuristic,
        batch_size=1000,
        random=False)

    loss = 0.
    num_corrects = 0.
    num_examples = 0
    for index, inputs in enumerate(batch_dataloader):
        batch_size = inputs["labels"].shape[0]
        batch_preds, batch_label_ids, batch_mean_loss = misc_utils.predict(
            trainer=trainer,
            model=model,
            inputs=inputs)

        num_examples += batch_size
        loss += batch_mean_loss * batch_size
        num_corrects += (batch_preds.argmax(axis=-1) == batch_label_ids).sum()

    return loss / num_examples, num_corrects / num_examples


def create_FAISS_index(
    train_task_name: str,
    trained_on_task_name: str,
) -> faiss_utils.FAISSIndex:
    if train_task_name not in ["mnli-2", "hans", "amazon", "anli"]:
        raise ValueError

    if trained_on_task_name not in ["mnli", "mnli-2", "hans", "amazon"]:
        raise ValueError

    if trained_on_task_name == "mnli":
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.MNLI_MODEL_PATH)

    if trained_on_task_name == "mnli-2":
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.MNLI2_MODEL_PATH)

    if trained_on_task_name == "hans":
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.HANS_MODEL_PATH)

    if trained_on_task_name == "amazon":
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.Amazon_MODEL_PATH)

    train_dataset, _ = misc_utils.create_datasets(
        task_name=train_task_name,
        tokenizer=tokenizer)

    faiss_index = faiss_utils.FAISSIndex(768, "Flat")

    model.cuda()
    device = model.device
    train_batch_data_loader = misc_utils.get_dataloader(
        dataset=train_dataset,
        batch_size=128,
        random=False)

    for inputs in tqdm(train_batch_data_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        features = misc_utils.compute_BERT_CLS_feature(model, **inputs)
        features = features.cpu().detach().numpy()
        faiss_index.add(features)

    return faiss_index
