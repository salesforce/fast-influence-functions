import os
import time
import torch
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import transformers
from tqdm import tqdm
from glob import glob
from copy import deepcopy
from contexttimer import Timer
from collections import defaultdict
from transformers import TrainingArguments
from transformers import default_data_collator
from typing import List, Dict, Tuple, Optional, Union, Any

from experiments import constants
from experiments import mnli_utils
from experiments import misc_utils
from experiments import remote_utils
from experiments import hans
from influence_utils import faiss_utils
from influence_utils import nn_influence_utils
from experiments.data_utils import (
    glue_output_modes,
    glue_compute_metrics)

MNLI_TRAINING_SCRIPT_NAME = "scripts/run_MNLI.20200913.sh"
NUM_DATAPOINTS_TO_REMOVE_CHOICES = [1, 100, 10000]

CORRECT_INDICES = sorted([
    # e.g., `KNN-recall.only-correct.50.0.pth.g0301.ll.unc.edu`
    int(f.split("/")[-1].split(".")[3])
    for f in glob(os.path.join(
        constants.MNLI_RETRAINING_INFLUENCE_OUTPUT_BASE_DIR,
        "*only-correct*")
    )
])
INCORRECT_INDICES = sorted([
    # e.g., `KNN-recall.only-correct.50.0.pth.g0301.ll.unc.edu`
    int(f.split("/")[-1].split(".")[3])
    for f in glob(os.path.join(
        constants.MNLI_RETRAINING_INFLUENCE_OUTPUT_BASE_DIR,
        "*only-incorrect*")
    )
])


def run_retraining_main(
        mode: str,
        num_examples_to_test: int):

    if mode not in ["full", "KNN-1000", "KNN-10000", "random"]:
        raise ValueError(f"Unrecognized `mode` {mode}")

    for example_relative_index in range(num_examples_to_test):
        for correct_mode in ["correct", "incorrect"]:
            if correct_mode == "correct":
                example_index = CORRECT_INDICES[example_relative_index]
            if correct_mode == "incorrect":
                example_index = INCORRECT_INDICES[example_relative_index]

            if mode in ["full", "KNN-1000", "KNN-10000"]:
                # Load file from local or sync from remote
                if mode == "full":
                    file_name = os.path.join(
                        constants.MNLI_RETRAINING_INFLUENCE_OUTPUT_BASE_DIR,
                        f"KNN-recall.only-{correct_mode}.50.{example_index}"
                        f".pth.g0301.ll.unc.edu")

                influences_dict = torch.load(file_name)
                if example_index != influences_dict["test_index"]:
                    raise ValueError

                if (correct_mode == "correct" and
                        influences_dict["correct"] is not True or
                        correct_mode == "incorrect" and
                        influences_dict["correct"] is True):
                    raise ValueError

                helpful_indices = misc_utils.sort_dict_keys_by_vals(
                    influences_dict["influences"])
                harmful_indices = helpful_indices[::-1]
                indices_dict = {
                    "helpful": helpful_indices,
                    "harmful": harmful_indices}

            if mode == "random":
                # Get indices corresponding to each label
                label_to_indices = mnli_utils.get_label_to_indices_map()
                np.random.shuffle(label_to_indices["neutral"])
                np.random.shuffle(label_to_indices["entailment"])
                np.random.shuffle(label_to_indices["contradiction"])
                indices_dict = {
                    "neutral": label_to_indices["neutral"],
                    "entailment": label_to_indices["entailment"],
                    "contradiction": label_to_indices["contradiction"],
                }

            for tag, indices in indices_dict.items():
                for num_data_points_to_remove in NUM_DATAPOINTS_TO_REMOVE_CHOICES:
                    run_one_retraining(
                        indices=indices[:num_data_points_to_remove],
                        dir_name=(
                            f"./retraining-remove-"
                            f"{example_index}-"
                            f"{correct_mode}-"
                            f"{mode}-"
                            f"{tag}-"
                            f"{num_data_points_to_remove}"))


def run_one_retraining(
        indices: List[int],
        dir_name: str,
) -> None:
    mnli_utils.create_one_set_of_data_for_retraining(
        dir_name=dir_name,
        indices_to_remove=indices)
    output_dir = os.path.join(dir_name, "output_dir")
    subprocess.check_call([
        "bash",
        MNLI_TRAINING_SCRIPT_NAME,
        dir_name, output_dir
    ])
    client = remote_utils.ScpClient()
    client.scp_file_to_remote(
        local_file_name=dir_name,
        remote_file_name=os.path.join(
            constants.REMOTE_DEFAULT_REMOTE_BASE_DIR,
            f"{dir_name}.{client.host_name}"),
        # This is a folder
        recursive=True)


def run_full_influence_functions(
        mode: str,
        num_examples_to_test: int,
        s_test_num_samples: int = 1000
) -> Dict[int, Dict[str, Any]]:

    if mode not in ["only-correct", "only-incorrect"]:
        raise ValueError(f"Unrecognized mode {mode}")

    tokenizer, model = misc_utils.create_tokenizer_and_model(
        constants.MNLI_MODEL_PATH)

    (mnli_train_dataset,
     mnli_eval_dataset) = misc_utils.create_datasets(
        task_name="mnli",
        tokenizer=tokenizer)

    batch_train_data_loader = misc_utils.get_dataloader(
        mnli_train_dataset,
        batch_size=128,
        random=True)

    instance_train_data_loader = misc_utils.get_dataloader(
        mnli_train_dataset,
        batch_size=1,
        random=False)

    eval_instance_data_loader = misc_utils.get_dataloader(
        dataset=mnli_eval_dataset,
        batch_size=1,
        random=False)

    output_mode = glue_output_modes["mnli"]

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
        model=model,
        args=TrainingArguments(
            output_dir="./tmp-output",
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            learning_rate=5e-5,
            logging_steps=100),
        data_collator=default_data_collator,
        train_dataset=mnli_train_dataset,
        eval_dataset=mnli_eval_dataset,
        compute_metrics=build_compute_metrics_fn("mnli"),
    )

    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    model.cuda()
    num_examples_tested = 0
    outputs_collections = {}
    for test_index, test_inputs in enumerate(eval_instance_data_loader):
        if num_examples_tested >= num_examples_to_test:
            break

        # Skip when we only want cases of correction prediction but the
        # prediction is incorrect, or vice versa
        prediction_is_correct = misc_utils.is_prediction_correct(
            trainer=trainer,
            model=model,
            inputs=test_inputs)

        if mode == "only-correct" and prediction_is_correct is False:
            continue

        if mode == "only-incorrect" and prediction_is_correct is True:
            continue

        with Timer() as timer:
            influences, _, s_test = nn_influence_utils.compute_influences(
                n_gpu=1,
                device=torch.device("cuda"),
                batch_train_data_loader=batch_train_data_loader,
                instance_train_data_loader=instance_train_data_loader,
                model=model,
                test_inputs=test_inputs,
                params_filter=params_filter,
                weight_decay=constants.WEIGHT_DECAY,
                weight_decay_ignores=weight_decay_ignores,
                s_test_damp=5e-3,
                s_test_scale=1e4,
                s_test_num_samples=s_test_num_samples,
                train_indices_to_include=None,
                s_test_iterations=1,
                precomputed_s_test=None)

            outputs = {
                "test_index": test_index,
                "influences": influences,
                "s_test": s_test,
                "time": timer.elapsed,
                "correct": prediction_is_correct,
            }
            num_examples_tested += 1
            outputs_collections[test_index] = outputs

            remote_utils.save_and_mirror_scp_to_remote(
                object_to_save=outputs,
                file_name=f"KNN-recall.{mode}.{num_examples_to_test}.{test_index}.pth")
            print(f"Status: #{test_index} | {num_examples_tested} / {num_examples_to_test}")

    return outputs_collections


def get_influences(
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


def main(
    label_list: List[str],
    task_model: torch.nn.Module,
    imitator_model: torch.nn.Module,
    trainer: transformers.Trainer,
    test_data_point_indices: List[int],
    batch_train_data_loader: torch.utils.data.DataLoader,
    instance_train_data_loader: torch.utils.data.DataLoader,
    instance_eval_data_loader: torch.utils.data.DataLoader,
    sample_size: int = 10,
    num_nearest_neighbors: int = 10000,
    finetune_using_ground_truth_label: bool = False
) -> List[Dict[str, Any]]:

    train_inputs_collections = torch.load(
        constants.MNLI_TRAIN_INPUT_COLLECTIONS_PATH)

    neutral_examples = []
    entailment_examples = []
    contradiction_examples = []
    for i in range(len(train_inputs_collections)):
        label = label_list[train_inputs_collections[i]["labels"]]
        if label == "neutral":
            neutral_examples.append(i)
        if label == "entailment":
            entailment_examples.append(i)
        if label == "contradiction":
            contradiction_examples.append(i)

    outputs_collections = []
    for i, test_inputs in enumerate(instance_eval_data_loader):
        if i not in test_data_point_indices:
            continue

        start_time = time.time()
        imitator_test_inputs = experimental_make_imitator_inputs(
            trainer=trainer, task_model=task_model, inputs=test_inputs)
        # if labels[0] != logits.argmax(axis=1)[0]:
        #     break
        influences, _ = get_influences(
            k=num_nearest_neighbors,
            model=task_model,
            test_inputs=test_inputs,
            batch_train_data_loader=batch_train_data_loader,
            instance_train_data_loader=instance_train_data_loader)

        data_indices = (
            np.random.choice(neutral_examples,
                             size=sample_size,
                             replace=False).tolist() +  # noqa
            np.random.choice(entailment_examples,
                             size=sample_size,
                             replace=False).tolist() +  # noqa
            np.random.choice(contradiction_examples,
                             size=sample_size,
                             replace=False).tolist() +  # noqa
            misc_utils.sort_dict_keys_by_vals(influences)[:sample_size] +  # noqa
            misc_utils.sort_dict_keys_by_vals(influences)[-sample_size:]
        )

        data_tags = (
            ["random-neutral" for _ in range(sample_size)] +  # noqa
            ["random-entailment" for _ in range(sample_size)] +  # noqa
            ["random-contradiction" for _ in range(sample_size)] +  # noqa
            ["most-negative-influential" for _ in range(sample_size)] +  # noqa
            ["most-positive-influential" for _ in range(sample_size)]
        )

        learning_rates = np.logspace(-5, -2.5, 50)
        losses = compute_new_imitator_losses(
            trainer=trainer,
            tags=data_tags,
            indices=data_indices,
            task_model=task_model,
            imitator_model=imitator_model,
            learning_rates=learning_rates,
            imitator_test_inputs=imitator_test_inputs,
            train_inputs_collections=train_inputs_collections,
            finetune_using_ground_truth_label=finetune_using_ground_truth_label)

        outputs_collections.append({
            "index": i,
            "losses": losses,
            "influences": influences,
            "test_inputs": test_inputs,
            "learning_rates": learning_rates,
            "imitator_test_inputs": imitator_test_inputs
        })

        end_time = time.time()
        print(f"#{len(outputs_collections)}/{len(outputs_collections)}: "
              f"Elapsed {(end_time - start_time) / 60:.2f}")

    return outputs_collections


def compute_new_imitator_losses(
        indices: List[int],
        tags: List[str],
        task_model: torch.nn.Module,
        imitator_model: torch.nn.Module,
        trainer: transformers.Trainer,
        learning_rates: Union[np.ndarray, List[float]],
        imitator_test_inputs: Dict[str, torch.Tensor],
        train_inputs_collections: List[Dict[str, torch.Tensor]],
        finetune_using_ground_truth_label: bool = False,
) -> Dict[str, List[List[float]]]:

    params_filter = [
        n for n, p in imitator_model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in imitator_model.named_parameters()
        if not p.requires_grad]

    losses = defaultdict(list)
    for index, tag in zip(tqdm(indices), tags):
        if finetune_using_ground_truth_label is True:
            imitator_train_inputs = train_inputs_collections[index]
        else:
            imitator_train_inputs = experimental_make_imitator_inputs(
                trainer=trainer,
                task_model=task_model,
                inputs=train_inputs_collections[index])

        _losses = []
        gradients_z = None
        for lr in learning_rates:
            # Re-use `gradients_z`
            new_imitator_model, gradients_z = hans.pseudo_gradient_step(
                model=imitator_model,
                inputs=imitator_train_inputs,
                learning_rate=lr,
                params_filter=params_filter,
                weight_decay_ignores=weight_decay_ignores,
                precomputed_gradients_z=gradients_z)
            _, _, imitator_loss = misc_utils.predict(
                trainer=trainer,
                model=new_imitator_model,
                inputs=imitator_test_inputs)
            _losses.append(imitator_loss)

        losses[tag].append(_losses)

    return losses


def experimental_make_imitator_inputs(
        trainer: transformers.Trainer,
        task_model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    logits, _, _ = misc_utils.predict(
        trainer=trainer, model=task_model, inputs=inputs)
    imitator_inputs = deepcopy(inputs)
    imitator_inputs["labels"] = torch.tensor(logits.argmax(axis=1))
    return imitator_inputs


def plot_Xs_and_Ys_dict(
        Xs: List[float],
        Ys_dict: Dict[str, List[List[float]]]
) -> None:
    # plt.rcParams["figure.figsize"] = (10, 10)
    color_map = {
        "random-neutral": "grey",
        "random-entailment": "salmon",
        "random-contradiction": "skyblue",
        "most-positive-influential": "darkred",
        "most-negative-influential": "steelblue"}

    legends = []
    for tag in Ys_dict.keys():
        if tag not in color_map.keys():
            raise ValueError

        legends.append(tag)
        color = color_map[tag]
        data = np.array(Ys_dict[tag])
        is_random_data_point = "random" in tag

        if data.shape[0] != 1:
            data_mean = data.mean(axis=0)
            data_max = data.max(axis=0)
            data_min = data.min(axis=0)
            # data_std = data.std(axis=0)
            plt.plot(Xs, data_mean,
                     color=color,
                     linestyle=("--" if is_random_data_point else None))

            # plt.fill_between(Xs,
            #                  data_mean + 1. * data_std,
            #                  data_mean - 1. * data_std,
            #                  color=color,
            #                  alpha=0.1 if is_random_data_point else 0.2)
            plt.fill_between(Xs, data_max, data_min,
                             alpha=0.1,
                             color=color)
        else:
            plt.plot(Xs, data[0, ...], color=color)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("learning rate", fontsize=30)
    plt.ylabel("Loss", fontsize=30)
    plt.legend(legends, fontsize=15)
    plt.title("Loss of the Imitator Model", fontsize=30)
    # plt.savefig("./20200719-fig1.pdf")
