import os
import torch
from typing import Any, Optional

from experiments import mnli
from experiments import s_test_speedup

EXPERIMENT_BASE_DIR = "/export/home/Experiments/20200901/"
NUM_KNN_RECALL_EXPERIMENTS = 50
NUM_STEST_EXPERIMENTS = 10


def save_object_to_file(object_to_save: Any, file_name: str) -> str:
    full_file_name = os.path.join(EXPERIMENT_BASE_DIR, file_name)
    torch.save(object_to_save, full_file_name)
    return full_file_name


def KNN_recall_experiments(num_experiments: Optional[int] = None) -> None:
    """Experiments to Check The Influence Recall of KNN"""
    if num_experiments is None:
        num_experiments = NUM_KNN_RECALL_EXPERIMENTS

    # (a) when the prediction is correct, and (b) incorrect
    correct_outputs_collections = mnli.run_full_influence_functions(
        mode="only-correct",
        num_examples_to_test=num_experiments)

    incorrect_outputs_collections = mnli.run_full_influence_functions(
        mode="only-incorrect",
        num_examples_to_test=num_experiments)



    save_object_to_file(
        object_to_save=incorrect_outputs_collections,
        file_name=f"KNN-recall-incorrect-{num_experiments}.pth")


def s_test_speed_quality_tradeoff_experiments(
        num_experiments: Optional[int] = None) -> None:
    """Experiments to Check The Speed/Quality Trade-off of `s_test` estimation"""
    if num_experiments is None:
        num_experiments = NUM_STEST_EXPERIMENTS

    # (a) when the prediction is correct, and (b) incorrect
    correct_outputs_collections = s_test_speedup.main(
        mode="only-correct",
        num_examples_to_test=num_experiments)

    incorrect_outputs_collections = s_test_speedup.main(
        mode="only-incorrect",
        num_examples_to_test=num_experiments)

    save_object_to_file(
        object_to_save=correct_outputs_collections,
        file_name=f"stest-correct-{num_experiments}.pth")

    save_object_to_file(
        object_to_save=incorrect_outputs_collections,
        file_name=f"stest-incorrect-{num_experiments}.pth")


def 