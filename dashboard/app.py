#yang ni latest
#writefile app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings

st.set_page_config(page_title="UrbanRent", page_icon="🏠")
filterwarnings('ignore', category=FutureWarning)

# Sidebar for navigation
#page = st.sidebar.radio("Navigate", ["Rental Prediction", "Interactive Graphs", "Data Analysis"])
# Sidebar navigation with icons
page = st.sidebar.radio(
    "",
    [
        "🏠 UrbanRent",
        "💲 Rent Prediction",
        "📊 Interactive Graphs",
        "🗺️ Map Visualisation",
        "🔍 Model Evaluation",
        #"Rent Duration"
    ]
)

if page == "🏠 UrbanRent":
    st.empty()
    exec(open("dashboard/home.py").read())

if page == "💲 Rent Prediction":
    st.empty()
    exec(open("dashboard/rental.py").read())

if page == "📊 Interactive Graphs":
    st.empty()
    exec(open("dashboard/dashboard.py").read())

if page == "🗺️ Map Visualisation":
    st.empty()
    exec(open("dashboard/visual.py").read())

if page == "🔍 Model Evaluation":
    st.empty()
    exec(open("dashboard/model.py").read())
