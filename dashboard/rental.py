import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings

# Load the best model
best_model_pipeline = joblib.load("model/best_model_pipeline.pkl")
linear_model_pipeline = joblib.load('model/linear_regression_pipeline.pkl')
gradient_boosting_pipeline = joblib.load('model/gradient_boosting_pipeline.pkl')

# Load the best model's name
with open("model/best_model_name.txt", "r") as f:
    best_model_name = f.read()

# Extract categories for user input from the best model's preprocessor
preprocessor = best_model_pipeline.named_steps['preprocessor']
onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
location_categories = onehot_encoder.categories_[0]
property_type_categories = onehot_encoder.categories_[1]

# Streamlit UI setup
st.title("Rental Price Prediction")
st.write(f"Predict monthly rental prices using the most accurate model: **{best_model_name}**")

# User input for property details
location = st.selectbox("Location", location_categories)
property_type = st.selectbox("Property Type", property_type_categories)
rooms = st.number_input("Number of Rooms", value=2)
bathrooms = st.number_input("Number of Bathrooms", value=1)
parking = st.number_input("Number of Parking Spaces", value=1)

# Rental duration with the selected options
#rental_duration = st.selectbox("Rental Duration (Months)", [12, 24, 36])

# Combine user inputs into a DataFrame
user_input = pd.DataFrame({
    'location': [location],
    'property_type': [property_type],
    'rooms': [rooms],
    'bathroom': [bathrooms],
    'parking': [parking]
})

# Function to adjust rental price based on duration
"""def get_price_adjustment_based_on_duration(base_price, duration):
    
    #if duration < 12:
        #adjusted_price = base_price * (1 + 0.10 * (12 - duration))  # Increase price for shorter durations
    if duration > 17:
        adjusted_price = base_price * (1 - 0.05)  # Decrease price for longer durations
    else:
        adjusted_price = base_price  # No adjustment for 12-month duration
    return adjusted_price
"""

# Predict with the best model
if st.button("Predict"):
    prediction = best_model_pipeline.predict(user_input)[0]
    linear_prediction = linear_model_pipeline.predict(user_input)[0]
    gradient_boosting_prediction = gradient_boosting_pipeline.predict(user_input)[0]

    """# Adjust price based on rental duration
    adjusted_prediction = get_price_adjustment_based_on_duration(prediction, rental_duration)
    adjusted_linear_prediction = get_price_adjustment_based_on_duration(linear_prediction, rental_duration)
    adjusted_gradient_boosting_prediction = get_price_adjustment_based_on_duration(gradient_boosting_prediction, rental_duration)
    """
    
    # Display predictions
    st.write(f"Predicted Rent using {best_model_name}: RM{prediction:.2f}")
    #st.write(f"Predicted Rent using Linear Regression: RM{linear_prediction:.2f}")
    #st.write(f"Predicted Rent using Gradient Boosting: RM{gradient_boosting_prediction:.2f}")

    #if (rental_duration > 17):
        #st.write(f"Adjusted Rent for {rental_duration} months: RM{adjusted_prediction:.2f}")
    #st.write(f"Adjusted Rent using Linear Regression for {rental_duration} months: RM{adjusted_linear_prediction:.2f}")
    #st.write(f"Adjusted Rent using Gradient Boosting for {rental_duration} months: RM{adjusted_gradient_boosting_prediction:.2f}")
