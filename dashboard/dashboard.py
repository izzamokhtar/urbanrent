#latesttt
%%writefile dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
from itertools import product

st.title("Rent Pricing at Kuala Lumpur and Selangor")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('/content/drive/MyDrive/FYP B IZZA/mudah-apartment-kl-selangor_cleaned.csv')
    df.columns = df.columns.str.strip().str.lower()
    df['monthly_rent'] = df['monthly_rent'].str.replace('RM', '').str.replace('per month', '').str.replace(' ', '')
    df['monthly_rent'] = pd.to_numeric(df['monthly_rent'], errors='coerce')
    df = df.dropna(subset=['monthly_rent'])
    return df

df = load_data()

# Load the trained model pipeline
@st.cache_resource
def load_model():
    model = joblib.load('/content/drive/MyDrive/FYP B IZZA/best_model_pipeline.pkl')  # Your trained model
    return model

model = load_model()

# Sidebar for user input (only Location and Property Type)
st.sidebar.title('Interactive Graphs')
st.sidebar.header('Graph Parameters')

# User selects the parameters for filtering data
selected_locations = st.sidebar.multiselect('Select Locations', df['location'].unique())
selected_property_types = st.sidebar.multiselect('Select Property Types', df['property_type'].unique())

# Ensure there is at least one location and one property type selected
if not selected_locations or not selected_property_types:
    st.warning("Please select at least one location and one property type.")
    st.stop()

# Generate all combinations of selected locations and property types
combinations = list(product(selected_locations, selected_property_types))

# Prepare input for the model
model_input = pd.DataFrame(combinations, columns=['location', 'property_type'])

# Add dummy values for other features (rooms, bathroom, parking)
model_input['rooms'] = 2  # Default room value
model_input['bathroom'] = 1  # Default bathroom value
model_input['parking'] = 1  # Default parking value

# Apply the same transformations as the model expects (e.g., one-hot encoding, scaling)
# Assuming the model uses a ColumnTransformer for preprocessing:
try:
    transformed_input = model.transform(model_input)
except AttributeError:
    # Handle the case where the model is not a pipeline and does not need transformation
    transformed_input = model_input  # You may need to manually apply transformations here

# Get predictions for rent prices from the model
predicted_prices = model.predict(transformed_input)

# Prepare the data for the graph (create a DataFrame with predicted prices)
graph_data = pd.DataFrame({
    'location': model_input['location'],
    'property_type': model_input['property_type'],
    'predicted_rent': predicted_prices
})

# Round the predicted prices to 2 decimal points
graph_data['predicted_rent'] = graph_data['predicted_rent'].round(2)

# Visualization Columns
col1, col2 = st.columns(2)

# Graph 1: Average Price by Location (based on model prediction)
avg_price_location = graph_data.groupby('location')['predicted_rent'].mean().reset_index()
avg_price_location['predicted_rent'] = avg_price_location['predicted_rent'].round(2)
fig1 = px.bar(avg_price_location, x='location', y='predicted_rent', color='location', title="Average Price by Location")

# Increase the graph size and rename axis labels
fig1.update_layout(
    title_font_size=20,  # Adjust title font size
    xaxis_title='Location',  # Rename the x-axis
    yaxis_title='Price (RM)',  # Rename the y-axis
    xaxis_title_font_size=14,  # Adjust x-axis title font size
    yaxis_title_font_size=14,  # Adjust y-axis title font size
    showlegend=True,  # Show legend if necessary
)

col1.plotly_chart(fig1, use_container_width=True)

# Graph 2: Average Price by Property Type (based on model prediction)
avg_price_property = graph_data.groupby('property_type')['predicted_rent'].mean().reset_index()
avg_price_property['predicted_rent'] = avg_price_property['predicted_rent'].round(2)
fig2 = px.bar(avg_price_property, x='property_type', y='predicted_rent', color='property_type', title="Average Price by Property Type")

# Increase the graph size and rename axis labels
fig2.update_layout(
    title_font_size=20,  # Adjust title font size
    xaxis_title='Property Type',  # Rename the x-axis
    yaxis_title='Price (RM)',  # Rename the y-axis
    xaxis_title_font_size=14,  # Adjust x-axis title font size
    yaxis_title_font_size=14,  # Adjust y-axis title font size
    showlegend=True,  # Show legend if necessary
)

col2.plotly_chart(fig2, use_container_width=True)
