#yang ni latest
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings

st.set_page_config(page_title="Rent Price Prediction", page_icon=":heavy_dollar_sign:")
filterwarnings('ignore', category=FutureWarning)

# Sidebar for navigation
#page = st.sidebar.radio("Navigate", ["Rental Prediction", "Interactive Graphs", "Data Analysis"])
# Sidebar navigation with icons
page = st.sidebar.radio(
    "",
    [
        "💲 Rent Prediction",
        "📊 Interactive Graphs",
        "📈 Map Visualisation"
    ]
)

if page == "💲 Rent Prediction":
    st.empty()
    exec(open("dashboard/rental.py").read())

if page == "📊 Interactive Graphs":
    st.empty()
    exec(open("dashboard/dashboard.py").read())

if page == "📈 Map Visualisation":
    st.empty()
    exec(open("dashboard/visual.py").read())
