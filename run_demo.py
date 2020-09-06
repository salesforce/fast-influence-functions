import streamlit as st
from experiments import demo_utils


st.title("Influence Demo")

with st.spinner("Loading Dataset"):
    data = demo_utils.load_dataset("mnli")

with st.spinner("Loading Model"):
    helper = demo_utils.DemoInfluenceHelper("mnli-2", "mnli-2")

data
number = st.number_input(
    "Insert a number",
    value=int(0),
    min_value=int(0),
    max_value=int(data.shape[0] - 1),
    format="%d")
if number > data.shape[0] or number < 0:
    raise ValueError("Invalid Number")
