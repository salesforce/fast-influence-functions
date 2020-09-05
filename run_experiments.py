from typing import Optional, Dict

from experiments import mnli
from experiments import s_test_speedup
from experiments import remote_utils

NUM_KNN_RECALL_EXPERIMENTS = 50
NUM_STEST_EXPERIMENTS = 10


def KNN_recall_experiments(
        num_experiments: Optional[int] = None
) -> None:
    """Experiments to Check The Influence Recall of KNN"""
    print("RUNNING `KNN_recall_experiments`")

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
    print("RUNNING `s_test_speed_quality_tradeoff_experiments`")

    if num_experiments is None:
        num_experiments = NUM_STEST_EXPERIMENTS

    # (a) when the prediction is correct, and (b) incorrect
    s_test_speedup.main(
        mode="only-correct",
        num_examples_to_test=num_experiments)

    s_test_speedup.main(
        mode="only-incorrect",
        num_examples_to_test=num_experiments)


# ------------------------------------------------------------------
# TEST FUNCTIONS
# ------------------------------------------------------------------
def check_KNN_recall_local_remote_match(
        local_output_collections: Dict,
        remote_output_collections: Dict
) -> None:
    assert local_output_collections["time"] == remote_output_collections["time"]
    assert local_output_collections["correct"] == remote_output_collections["correct"]
    assert local_output_collections["influences"] == remote_output_collections["influences"]
    for index in range(len(local_output_collections["s_test"])):
        assert (local_output_collections["s_test"][index] == remote_output_collections["s_test"][index]).all()


if __name__ == "__main__":
    # Make sure the environment is properly setup
    remote_utils.setup_and_verify_environment()
    KNN_recall_experiments()
