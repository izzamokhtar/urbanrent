#latest
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings

# Load the best model
best_model_pipeline = joblib.load(f"{getcwd()}/model/best_model_pipeline.pkl")
linear_model_pipeline = joblib.load(f'{getcwd()}/model/linear_regression_pipeline.pkl')
#random_forest_pipeline = joblib.load('/content/drive/MyDrive/FYP B IZZA/random_forest_pipeline.pkl')
gradient_boosting_pipeline = joblib.load(f'{getcwd()}model/gradient_boosting_pipeline.pkl')

# Load the best model's name
with open(f"{getcwd()}/model/best_model_name.txt", "r") as f:
    best_model_name = f.read()

# Extract categories for user input from the best model's preprocessor
preprocessor = best_model_pipeline.named_steps['preprocessor']
onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
location_categories = onehot_encoder.categories_[0]
property_type_categories = onehot_encoder.categories_[1]

st.title("Rental Price Prediction")
st.write(f"Predict monthly rental prices using the most accurate model: **{best_model_name}**")

# User input
location = st.selectbox("Location", location_categories)
property_type = st.selectbox("Property Type", property_type_categories)
rooms = st.number_input("Number of Rooms", value=2)
bathrooms = st.number_input("Number of Bathrooms", value=1)
parking = st.number_input("Number of Parking Spaces", value=1)

# Combine user inputs into a DataFrame
user_input = pd.DataFrame({
    'location': [location],
    'property_type': [property_type],
    'rooms': [rooms],
    'bathroom': [bathrooms],
    'parking': [parking]
})

# Predict with the best model
if st.button("Predict"):
    prediction = best_model_pipeline.predict(user_input)[0]
    st.write(f"Predicted Rent using {best_model_name}: RM{prediction:.2f}")
    linear_prediction = linear_model_pipeline.predict(user_input)[0]
    gradient_boosting_prediction = gradient_boosting_pipeline.predict(user_input)[0]

    # Plot predictions
    algorithms = ['Linear Regression', 'Random Forest', 'Gradient Boosting']
    predictions = [linear_prediction, prediction, gradient_boosting_prediction]

    # Create a bar chart
    fig, ax = plt.subplots()
    ax.bar(algorithms, predictions, color=['blue', 'green', 'orange'])
    ax.set_ylabel('Predicted Rental Price (RM)')
    ax.set_title('Comparison of Predicted Prices')
    st.pyplot(fig)
