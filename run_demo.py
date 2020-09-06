import streamlit as st
from experiments import demo_utils


st.title("Influence Demo")


def setup():
    with st.spinner("Loading Model and Dataset"):
        data = demo_utils.load_dataset("hans")
        helper = demo_utils.ExperimentalDemoInfluenceHelper(
            train_task_name="mnli-2",
            eval_task_name="hans",
            hans_heuristic="lexical_overlap")

    return data, helper


def run(index: int, helper: demo_utils.ExperimentalDemoInfluenceHelper):
    with st.spinner("Running Influence"):
        inputs, influences = helper.run(index)
    demo_utils.print_influential_examples(
        test_input=inputs,
        tokenizer=helper._tokenizer,
        influences=influences,
        train_dataset=helper._train_dataset,
        printer_fn=st.write)


data, helper = setup()


data
number = st.number_input(
    "Insert a number",
    value=int(0),
    min_value=int(0),
    max_value=int(data.shape[0] - 1),
    format="%d")
if number > data.shape[0] or number < 0:
    raise ValueError("Invalid Number")

run(number, helper)
