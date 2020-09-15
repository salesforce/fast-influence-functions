import os
import torch
import shutil
import pandas as pd
from transformers import (
    BertTokenizer,
    InputFeatures,
    default_data_collator)
from typing import Tuple, Optional, Union, List, Dict
from experiments import constants


def decode_one_example(
        tokenizer: BertTokenizer,
        label_list: List[str],
        inputs: Dict[str, torch.Tensor],
        logits: Optional[torch.FloatTensor] = None
) -> Union[Tuple[str, str], Tuple[str, str, str]]:

    if inputs["input_ids"].shape[0] != 1:
        raise ValueError

    X = tokenizer.decode(inputs["input_ids"][0])
    Y = label_list[inputs["labels"].item()]
    if logits is not None:
        _Y_hat = logits.argmax(dim=-1).item()
        Y_hat = label_list[_Y_hat]
        return X, Y, Y_hat
    else:
        return X, Y


def visualize(tokenizer: BertTokenizer,
              label_list: List[str],
              inputs: Dict[str, torch.Tensor],) -> None:
    X, Y = decode_one_example(
        tokenizer=tokenizer,
        label_list=label_list,
        inputs=inputs,
        logits=None)
    premise, hypothesis = X.split("[CLS]")[1].split("[SEP]")[:2]
    print(f"\tP: {premise.strip()}\n\tH: {hypothesis.strip()}\n\tL: {Y}")


def get_data_from_features_or_inputs(
        tokenizer: BertTokenizer,
        label_list: List[str],
        feature: Optional[InputFeatures] = None,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[str, str, str]:

    if feature is not None and inputs is None:
        inputs = default_data_collator([feature])

    elif feature is None and inputs is not None:
        pass

    elif feature is None and inputs is None:
        raise ValueError

    elif feature is not None and inputs is not None:
        raise ValueError

    X, Y = decode_one_example(
        tokenizer=tokenizer,
        label_list=label_list,
        inputs=inputs,
        logits=None)
    premise, hypothesis = X.split("[CLS]")[1].split("[SEP]")[:2]
    return premise.strip(), hypothesis.strip(), Y


def create_one_set_of_data_for_retraining(
        dir_name: str,
        indices_to_remove: List[int],
) -> None:
    """Create the training data and evaluation data

    1. Load the training data, remove lines based in inputs, and write.
    2. Copy the evaluation data into the same directory.

    """
    with open(constants.MNLI_TRAIN_FILE_NAME) as f:
        lines = f.readlines()

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    else:
        raise ValueError

    with open(os.path.join(dir_name, "train.tsv"), "w") as f:
        lines_to_write = [
            # "-1" because of the header line
            l for i, l in enumerate(lines)
            if i - 1 not in indices_to_remove]

        f.write("".join(lines_to_write))
        print(f"Wrote {len(lines_to_write)} to {dir_name}")

    shutil.copyfile(constants.MNLI_EVAL_MATCHED_FILE_NAME,
                    os.path.join(dir_name, "dev_matched.tsv"))
    shutil.copyfile(constants.MNLI_EVAL_MISMATCHED_FILE_NAME,
                    os.path.join(dir_name, "dev_mismatched.tsv"))


def get_label_to_indices_map() -> Dict[str, List[int]]:
    with open(constants.MNLI_TRAIN_FILE_NAME) as f:
        lines = f.readlines()

    data_frame = pd.DataFrame(
        [line.strip().split("\t") for line in lines[1:]],
        columns=lines[0].strip().split("\t"))

    return {
        "contradiction": (
            data_frame[data_frame.gold_label == "contradiction"].index),
        "entailment": (
            data_frame[data_frame.gold_label == "entailment"].index),
        "neutral": (
            data_frame[data_frame.gold_label == "neutral"].index),
    }


def get_label_to_indices_map_2() -> Dict[str, List[int]]:
    """Slower, deprecated"""
    contradiction_indices = []
    entailment_indices = []
    neutral_indices = []
    train_inputs_collections = torch.load(constants.MNLI_TRAIN_INPUT_COLLECTIONS_PATH)
    for index, train_inputs in enumerate(train_inputs_collections):
        if train_inputs["labels"].item() == 0:
            contradiction_indices.append(index)
        if train_inputs["labels"].item() == 1:
            entailment_indices.append(index)
        if train_inputs["labels"].item() == 2:
            neutral_indices.append(index)

    return {
        "contradiction": contradiction_indices,
        "entailment": entailment_indices,
        "neutral": neutral_indices,
    }
