# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import sys
from typing import Optional, Dict

from experiments import mnli
from experiments import hans
from experiments import s_test_speedup
from experiments import remote_utils
from experiments import visualization


USE_PARALLEL = False
NUM_KNN_RECALL_EXPERIMENTS = 50
NUM_RETRAINING_EXPERIMENTS = 3
NUM_STEST_EXPERIMENTS = 10
NUM_VISUALIZATION_EXPERIMENTS = 100
NUM_IMITATOR_EXPERIMENTS = 10


def KNN_recall_experiments(
        mode: str,
        num_experiments: Optional[int] = None
) -> None:
    """Experiments to Check The Influence Recall of KNN"""
    print("RUNNING `KNN_recall_experiments`")

    if num_experiments is None:
        num_experiments = NUM_KNN_RECALL_EXPERIMENTS

    # (a) when the prediction is correct, and (b) incorrect
    mnli.run_full_influence_functions(
        mode=mode,
        num_examples_to_test=num_experiments)


def s_test_speed_quality_tradeoff_experiments(
        mode: str,
        num_experiments: Optional[int] = None
) -> None:
    """Experiments to Check The Speed/Quality Trade-off of `s_test` estimation"""
    print("RUNNING `s_test_speed_quality_tradeoff_experiments`")

    if num_experiments is None:
        num_experiments = NUM_STEST_EXPERIMENTS

    # (a) when the prediction is correct, and (b) incorrect
    s_test_speedup.main(
        mode=mode,
        num_examples_to_test=num_experiments)


def MNLI_retraining_experiments(
        mode: str,
        num_experiments: Optional[int] = None
) -> None:
    print("RUNNING `MNLI_retraining_experiments`")

    if num_experiments is None:
        num_experiments = NUM_RETRAINING_EXPERIMENTS

    mnli.run_retraining_main(
        mode=mode,
        num_examples_to_test=num_experiments)


def visualization_experiments(
        num_experiments: Optional[int] = None
) -> None:
    """Experiments for Visualizing Effects"""
    print("RUNNING `visualization_experiments`")

    if num_experiments is None:
        num_experiments = NUM_VISUALIZATION_EXPERIMENTS

    for heuristic in hans.DEFAULT_EVAL_HEURISTICS:
        visualization.main(
            train_task_name="hans",
            eval_task_name="hans",
            num_eval_to_collect=num_experiments,
            use_parallel=USE_PARALLEL,
            hans_heuristic=heuristic,
            trained_on_task_name="hans")

    visualization.main(
        train_task_name="hans",
        eval_task_name="mnli-2",
        num_eval_to_collect=num_experiments,
        use_parallel=USE_PARALLEL,
        hans_heuristic=None,
        trained_on_task_name="hans")


def hans_augmentation_experiments(
        num_replicas: Optional[int] = None
) -> None:
    print("RUNNING `hans_augmentation_experiments`")
    # We will use the all the `train_heuristic` here, as we did in
    # `eval_heuristics`. So looping over the `DEFAULT_EVAL_HEURISTICS`
    for train_task_name in ["mnli-2", "hans"]:
        for train_heuristic in hans.DEFAULT_EVAL_HEURISTICS:
            for version in ["new-only-z", "new-only-ztest", "new-z-and-ztest"]:
                hans.main(
                    train_task_name=train_task_name,
                    train_heuristic=train_heuristic,
                    num_replicas=num_replicas,
                    use_parallel=USE_PARALLEL,
                    version=version)


def imitator_experiments(
        num_experiments: Optional[int] = None
) -> None:
    print("RUNNING `imitator_experiments`")

    if num_experiments is None:
        num_experiments = NUM_IMITATOR_EXPERIMENTS

    mnli.imitator_main(
        mode="only-correct",
        num_examples_to_test=num_experiments)

    mnli.imitator_main(
        mode="only-incorrect",
        num_examples_to_test=num_experiments)


if __name__ == "__main__":
    # Make sure the environment is properly setup
    remote_utils.setup_and_verify_environment()

    experiment_name = sys.argv[1]
    if experiment_name == "knn-recall-correct":
        KNN_recall_experiments(
            mode="only-correct")
    if experiment_name == "knn-recall-incorrect":
        KNN_recall_experiments(
            mode="only-incorrect")

    if experiment_name == "s-test-correct":
        s_test_speed_quality_tradeoff_experiments(
            mode="only-correct")
    if experiment_name == "s-test-incorrect":
        s_test_speed_quality_tradeoff_experiments(
            mode="only-incorrect")

    if experiment_name == "retraining-full":
        MNLI_retraining_experiments(
            mode="full")

    if experiment_name == "retraining-random":
        MNLI_retraining_experiments(
            mode="random")

    if experiment_name == "retraining-KNN-1000":
        MNLI_retraining_experiments(
            mode="KNN-1000")

    if experiment_name == "retraining-KNN-10000":
        MNLI_retraining_experiments(
            mode="KNN-10000")

    if experiment_name == "hans-augmentation":
        hans_augmentation_experiments()

    if experiment_name == "imitator":
        imitator_experiments()

    # raise ValueError(f"Unknown Experiment Name: {experiment_name}")
