import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset (replace with your dataset path)
def load_data():
    data = pd.read_csv('dataset/mudah-apartment-kl-selangor_cleaned.csv')
    data.columns = data.columns.str.strip().str.lower()
    data['monthly_rent'] = data['monthly_rent'].str.replace('RM', '').str.replace('per month', '').str.replace(' ', '')
    data['monthly_rent'] = pd.to_numeric(data['monthly_rent'], errors='coerce')
    data = data.dropna(subset=['monthly_rent'])
    return data

data = load_data()

st.title("UrbanRent: Predictive Pricing for Urban Rentals 💲")

# Sidebar Filters
st.sidebar.header("Filter Options")
location_filter = st.sidebar.multiselect(
    "Select Location", options=data['region'].unique(), default=data['region'].unique()
)
property_type_filter = st.sidebar.multiselect(
    "Select Property Type", options=data['property_type'].unique(), default=data['property_type'].unique()
)

# Apply Filters
filtered_data = data[
    (data['region'].isin(location_filter)) & (data['property_type'].isin(property_type_filter))
]

# Custom function for styled metric cards
def custom_metric(label, value, color):
    st.markdown(
        f"""
        <div style="
            background-color: {color}; 
            padding: 10px; 
            margin: 10px; 
            border-radius: 20px; 
            text-align: center; 
            color: white; 
            width: 220px; 
            height: 120px;
            display: flex; 
            flex-direction: column; 
            justify-content: center; 
            align-items: center;">
            <h3 style="margin: 0; font-size: 18px;">{label}</h3>
            <h1 style="margin: 0; font-size: 28px;">{value}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# Metrics Section
st.header(" ")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    avg_rent = f"RM {filtered_data['monthly_rent'].mean():,.2f}"
    custom_metric("Average Price", avg_rent, "#4CAF50")
with col2:
    total_properties = f"{len(filtered_data):,}"
    custom_metric("Total Properties", total_properties, "#2196F3")
with col3:
    most_common_type = filtered_data['property_type'].mode()[0]
    custom_metric("Most Common Property Type", most_common_type, "#FF5722")

# Property Type Distribution using Plotly
st.header("Property Type Distribution")
avg_rent_by_type = filtered_data.groupby('property_type')['monthly_rent'].mean().reset_index()

# Round the mean rental prices to 2 significant figures
avg_rent_by_type['monthly_rent'] = avg_rent_by_type['monthly_rent'].round(2)

# Renaming columns for better readability
avg_rent_by_type = avg_rent_by_type.rename(columns={'property_type': 'Property Type', 'monthly_rent': 'Monthly Rent'})

fig = px.pie(
    avg_rent_by_type, 
    values='Monthly Rent', 
    names='Property Type', 
    title="Average Monthly Rent by Property Type", 
    color_discrete_sequence=px.colors.sequential.RdBu,
    hover_data={'Property Type': True, 'Monthly Rent': True}
)
# Customize the hover template to display the values clearly
fig.update_traces(
    hovertemplate='<b>%{label}</b><br>Monthly Rent: RM %{value}</br>'
)
st.plotly_chart(fig)

# Average Rent by Region using Plotly
st.header("Average Rent by Region")
avg_rent_by_region = filtered_data.groupby("region")["monthly_rent"].mean().reset_index()
fig = px.bar(
    avg_rent_by_region, 
    x="region", 
    y="monthly_rent", 
    title="Average Rental Price by Region", 
    color='region',
    color_continuous_scale='Viridis',
    labels={"region": "Region", "monthly_rent": "Average Monthly Rent (RM)"}
)
st.plotly_chart(fig)

# Size vs Rent Relationship using Plotly
st.header("Size vs Rent")
fig = px.scatter(
    filtered_data, 
    x="size", 
    y="monthly_rent", 
    title="Size vs Monthly Rent", 
    color='property_type', 
    hover_data=['region', 'property_type'],
    color_discrete_sequence=px.colors.qualitative.Set1,
    labels={"size": "Size", "monthly_rent": "Average Monthly Rent (RM)", "region": "Region", "property_type": "Property Type"}
)
st.plotly_chart(fig)

# Display Filtered Data (Optional)
#st.header("Filtered Data")
#st.dataframe(filtered_data)
