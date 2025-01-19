import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pandas as pd

#st.title("Model accuracy comparison")

# Load models
models = {
    'Linear Regression': joblib.load('model/linear_regression_model.pkl'),
    'Random Forest': joblib.load('model/random_forest_model.pkl'),
    'Gradient Boosting': joblib.load('model/gradient_boosting_model.pkl'),
}

# Example X_test and y_test (replace these with your actual data)
X_test = pd.read_csv('dataset/X_test.csv')  # Replace with actual test features
y_test = pd.read_csv('dataset/y_test.csv')  # Replace with actual test labels

# Function to compute metrics
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return mae, mse, rmse, r2

# Streamlit Page for Model Results
def model_results_page():
    st.title("Model Evaluation Metrics")

    # Multi-select to choose models for evaluation
    selected_models = st.multiselect("Select Models", list(models.keys()))

    if len(selected_models) == 0:
        st.warning("Please select at least one model.")
        return

    # Initialize empty lists to store metric values for each model
    mae_values, mse_values, rmse_values, r2_values = [], [], [], []

    # Loop through each selected model and evaluate
    for model_name in selected_models:
        model = models[model_name]
        mae, mse, rmse, r2 = evaluate_model(model, X_test, y_test)

        # Append the results for each model
        mae_values.append(mae)
        mse_values.append(mse)
        rmse_values.append(rmse)
        r2_values.append(r2)

    # Data for Plotly charts (4 separate charts for each metric)
    metrics_data = {
        'Model': selected_models,
        'MAE': mae_values,
        'MSE': mse_values,
        'RMSE': rmse_values,
        'R²': r2_values
    }

    # Define custom colors for each model
    model_colors = {
        'Linear Regression': 'blue',
        'Random Forest': 'green',
        'Gradient Boosting': 'red'
    }

    # Create bar charts for each metric
    st.subheader("Mean Absolute Error (MAE)")
    mae_fig = px.bar(
        metrics_data,
        x="Model",
        y="MAE",
        title="Mean Absolute Error (MAE) Comparison",
        text="MAE",
        labels={"Model": "Model", "MAE": "Score"},
        color='Model',
        color_discrete_map=model_colors
    )
    mae_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(mae_fig, use_container_width=True)

    st.subheader("Mean Squared Error (MSE)")
    mse_fig = px.bar(
        metrics_data,
        x="Model",
        y="MSE",
        title="Mean Squared Error (MSE) Comparison",
        text="MSE",
        labels={"Model": "Model", "MSE": "Score"},
        color='Model',
        color_discrete_map=model_colors
    )
    mse_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(mse_fig, use_container_width=True)

    st.subheader("Root Mean Squared Error (RMSE)")
    rmse_fig = px.bar(
        metrics_data,
        x="Model",
        y="RMSE",
        title="Root Mean Squared Error (RMSE) Comparison",
        text="RMSE",
        labels={"Model": "Model", "RMSE": "Score"},
        color='Model',
        color_discrete_map=model_colors
    )
    rmse_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(rmse_fig, use_container_width=True)

    st.subheader("R² Score")
    r2_fig = px.bar(
        metrics_data,
        x="Model",
        y="R²",
        title="R² Score Comparison",
        text="R²",
        labels={"Model": "Model", "R²": "Score"},
        color='Model',
        color_discrete_map=model_colors
    )
    r2_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(r2_fig, use_container_width=True)

# Call the function to render the page
model_results_page()
