import streamlit as st
from experiments import demo_utils


st.title("Influence Demo")


def setup():
    with st.spinner("Loading Model and Dataset"):
        data = demo_utils.load_dataset("mnli")
        helper = demo_utils.DemoInfluenceHelper("mnli-2", "mnli-2")

    return data, helper


def run(index: int, helper: demo_utils.DemoInfluenceHelper):
    with st.spinner("Running Influence"):
        influences = helper.run(index)
    demo_utils.print_most_influential_examples(
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
