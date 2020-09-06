import streamlit as st
from experiments import demo_utils


st.title("Influence Demo")
data = demo_utils.load_dataset("mnli")
data
