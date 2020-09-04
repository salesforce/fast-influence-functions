import os
from typing import Optional
from transformers import trainer_utils

from experiments import mnli
from experiments import s_test_speedup

NUM_KNN_RECALL_EXPERIMENTS = 50
NUM_STEST_EXPERIMENTS = 10


def KNN_recall_experiments(
        num_experiments: Optional[int] = None
) -> None:
    """Experiments to Check The Influence Recall of KNN"""
    if num_experiments is None:
        num_experiments = NUM_KNN_RECALL_EXPERIMENTS

    # (a) when the prediction is correct, and (b) incorrect
    mnli.run_full_influence_functions(
        mode="only-correct",
        num_examples_to_test=num_experiments)

    mnli.run_full_influence_functions(
        mode="only-incorrect",
        num_examples_to_test=num_experiments)


def s_test_speed_quality_tradeoff_experiments(
        num_experiments: Optional[int] = None
) -> None:
    """Experiments to Check The Speed/Quality Trade-off of `s_test` estimation"""
    if num_experiments is None:
        num_experiments = NUM_STEST_EXPERIMENTS

    # (a) when the prediction is correct, and (b) incorrect
    s_test_speedup.main(
        mode="only-correct",
        num_examples_to_test=num_experiments)

    s_test_speedup.main(
        mode="only-incorrect",
        num_examples_to_test=num_experiments)


def setup_and_verify_environment():
    # Check the environment
    if os.getenv("REMOTE_BASE_DIR") is None:
        raise ValueError(f"`REMOTE_BASE_DIR` is not set.")

    if trainer_utils.is_wandb_available() is False:
        raise ValueError("Weight And Bias is not set.")
