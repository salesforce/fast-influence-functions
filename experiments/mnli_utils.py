import torch
from transformers import (
    BertTokenizer,
    InputFeatures,
    default_data_collator)
from typing import Tuple, Optional, Union, List, Dict


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
