import torch
import numpy as np
import transformers
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from transformers import default_data_collator
from typing import Union, Dict, Any, List, Tuple, Optional

from influence_utils import parallel
from influence_utils import faiss_utils
from influence_utils import nn_influence_utils
from experiments import constants
from experiments import misc_utils
from experiments import remote_utils
from experiments.hans_utils import HansHelper
from transformers import TrainingArguments
from experiments.data_utils import (
    glue_output_modes,
    glue_compute_metrics,
    CustomGlueDataset)

DEFAULT_KNN_K = 1000
DEFAULT_NUM_REPLICAS = 3
EXPERIMENT_TYPES = ["most-helpful", "most-harmful", "random"]
DEFAULT_EVAL_HEURISTICS = ["lexical_overlap", "subsequence", "constituent"]


def main(
        train_heuristic: str,
        eval_heuristics: Optional[List[str]] = None,
        num_replicas: Optional[int] = None,
        use_parallel: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:

    if eval_heuristics is None:
        eval_heuristics = DEFAULT_EVAL_HEURISTICS

    if num_replicas is None:
        num_replicas = DEFAULT_NUM_REPLICAS

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

    hans_helper = HansHelper(
        hans_train_dataset=hans_train_dataset,
        hans_eval_dataset=hans_eval_dataset)

    # We will be running model trained on MNLI-2
    # but calculate influences on HANS dataset
    faiss_index = faiss_utils.FAISSIndex(768, "Flat")
    faiss_index.load(constants.MNLI2_HANS_FAISS_INDEX_PATH)

    output_mode = glue_output_modes["mnli-2"]

    def build_compute_metrics_fn(task_name: str):
        def compute_metrics_fn(p):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

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
        data_collator=default_data_collator,
        train_dataset=mnli_train_dataset,
        eval_dataset=hans_eval_dataset,
        compute_metrics=build_compute_metrics_fn("mnli-2"),
    )

    params_filter = [
        n for n, p in task_model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in task_model.named_parameters()
        if not p.requires_grad]

    output_collections = defaultdict(list)
    with tqdm(total=len(EXPERIMENT_TYPES) * num_replicas) as pbar:
        for experiment_type in EXPERIMENT_TYPES:
            for replica_index in range(num_replicas):
                outputs_one_experiment = one_experiment(
                    use_parallel=use_parallel,
                    train_heuristic=train_heuristic,
                    eval_heuristics=eval_heuristics,
                    experiment_type=experiment_type,
                    hans_helper=hans_helper,
                    hans_train_dataset=hans_train_dataset,
                    task_model=task_model,
                    faiss_index=faiss_index,
                    params_filter=params_filter,
                    weight_decay_ignores=weight_decay_ignores,
                    trainer=trainer)
                output_collections[experiment_type].append(outputs_one_experiment)

                pbar.update(1)
                pbar.set_description(f"{experiment_type} #{replica_index}")

    remote_utils.save_and_mirror_scp_to_remote(
        object_to_save=output_collections,
        file_name=f"hans-augmentation.{train_heuristic}.{num_replicas}.pth")

    return output_collections


def one_experiment(
    use_parallel: bool,
    train_heuristic: str,
    eval_heuristics: List[str],
    experiment_type: str,
    hans_helper: HansHelper,
    hans_train_dataset: CustomGlueDataset,
    task_model: torch.nn.Module,
    faiss_index: faiss_utils.FAISSIndex,
    params_filter: List[str],
    weight_decay_ignores: List[str],
    trainer: transformers.Trainer,
) -> Dict[str, Any]:
    if task_model.device.type != "cuda":
        raise ValueError("The model is supposed to be on CUDA")

    if experiment_type in ["most-harmful", "most-helpful"]:

        hans_eval_heuristic_inputs = hans_helper.sample_batch_of_heuristic(
            mode="eval", heuristic=train_heuristic, size=128)

        misc_utils.move_inputs_to_device(
            inputs=hans_eval_heuristic_inputs,
            device=task_model.device)

        if faiss_index is not None:
            features = misc_utils.compute_BERT_CLS_feature(
                task_model, **hans_eval_heuristic_inputs)
            features = features.cpu().detach().numpy()
            # We use the mean embedding as the final query here
            features = features.mean(axis=0, keepdims=True)
            KNN_distances, KNN_indices = faiss_index.search(
                k=DEFAULT_KNN_K, queries=features)
        else:
            KNN_indices = None

        if not use_parallel:
            batch_train_data_loader = misc_utils.get_dataloader(
                hans_train_dataset,
                batch_size=1,
                random=True)

            instance_train_data_loader = misc_utils.get_dataloader(
                hans_train_dataset,
                batch_size=1,
                random=False)

            influences, _, _ = nn_influence_utils.compute_influences(
                n_gpu=1,
                device=torch.device("cuda"),
                batch_train_data_loader=batch_train_data_loader,
                instance_train_data_loader=instance_train_data_loader,
                model=task_model,
                test_inputs=hans_eval_heuristic_inputs,
                params_filter=params_filter,
                weight_decay=constants.WEIGHT_DECAY,
                weight_decay_ignores=weight_decay_ignores,
                s_test_damp=5e-3,
                s_test_scale=1e6,
                s_test_num_samples=1000,
                train_indices_to_include=KNN_indices,
                precomputed_s_test=None)
        else:
            influences, s_test = parallel.compute_influences_parallel(
                # Avoid clash with main process
                device_ids=[1, 2, 3],
                train_dataset=hans_train_dataset,
                batch_size=1,
                model=task_model,
                test_inputs=hans_eval_heuristic_inputs,
                params_filter=params_filter,
                weight_decay=constants.WEIGHT_DECAY,
                weight_decay_ignores=weight_decay_ignores,
                s_test_damp=5e-3,
                s_test_scale=1e6,
                s_test_num_samples=1000,
                train_indices_to_include=KNN_indices,
                return_s_test=True,
                debug=False)

        sorted_indices = misc_utils.sort_dict_keys_by_vals(influences)
        if experiment_type == "most-helpful":
            datapoint_indices = sorted_indices

        if experiment_type == "most-harmful":
            # So that `datapoint_indices[:n]` return the
            # top-n most harmful datapoints
            datapoint_indices = sorted_indices[::-1]

    if experiment_type == "random":
        s_test = None
        influences = None
        hans_eval_heuristic_inputs = None
        # Essentially shuffle the indices
        datapoint_indices = np.random.choice(
            len(hans_train_dataset),
            size=len(hans_train_dataset),
            replace=False)

    loss_collections = {}
    accuracy_collections = {}
    num_datapoints_choices = [1, 10, 100]
    learning_rate_choices = [1e-5, 1e-4, 1e-3]
    for num_datapoints in num_datapoints_choices:
        for learning_rate in learning_rate_choices:
            datapoints = [
                hans_train_dataset[index]
                for index in datapoint_indices[:num_datapoints]]
            batch = default_data_collator(datapoints)
            new_model = pseudo_gradient_step(
                model=task_model,
                inputs=batch,
                learning_rate=learning_rate,
                params_filter=params_filter,
                weight_decay_ignores=weight_decay_ignores)

            for heuristic in eval_heuristics:
                new_model_loss, new_model_accuracy = evaluate_heuristic(
                    hans_helper=hans_helper,
                    heuristic=heuristic,
                    trainer=trainer,
                    model=new_model)

                loss_collections[
                    f"{num_datapoints}-"
                    f"{learning_rate}-"
                    f"{heuristic}"] = new_model_loss

                accuracy_collections[
                    f"{num_datapoints}-"
                    f"{learning_rate}-"
                    f"{heuristic}"] = new_model_accuracy
                # print(f"Finished {num_datapoints}-{learning_rate}")

    output_collections = {
        "s_test": s_test,
        "influences": influences,
        "loss": loss_collections,
        "accuracy": accuracy_collections,
        "datapoint_indices": datapoint_indices,
        "learning_rates": learning_rate_choices,
        "num_datapoints": num_datapoints_choices,
        "hans_eval_heuristic_inputs": hans_eval_heuristic_inputs,
    }
    return output_collections


def pseudo_gradient_step(
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        learning_rate: float,
        params_filter: List[str],
        weight_decay_ignores: List[str],
) -> torch.nn.Module:

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

    with torch.no_grad():
        [p.sub_(learning_rate * grad_z) for p, grad_z in
         zip(params_to_update, gradients_z)]

    return new_model


def evaluate_heuristic(
        hans_helper: HansHelper,
        heuristic: str,
        trainer: transformers.Trainer,
        model: torch.nn.Module,
) -> Tuple[float, float]:

    _, batch_dataloader = hans_helper.get_dataset_and_dataloader_of_heuristic(
        mode="eval",
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
    if train_task_name not in ["mnli-2", "hans"]:
        raise ValueError

    if trained_on_task_name not in ["mnli-2", "hans"]:
        raise ValueError

    if trained_on_task_name == "mnli-2":
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.MNLI2_MODEL_PATH)

    if trained_on_task_name == "hans":
        tokenizer, model = misc_utils.create_tokenizer_and_model(
            constants.HANS_MODEL_PATH)

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
