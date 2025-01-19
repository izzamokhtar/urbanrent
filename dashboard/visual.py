# visual.py
import streamlit as st
import pandas as pd
import folium  # type: ignore
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from folium.plugins import MarkerCluster  # type: ignore
import joblib
import time

st.title("Map Visualisation")

# Load your dataset
@st.cache_data
def load_data():
    # Replace with your actual dataset path
    df = pd.read_csv('/workspaces/urbanrent/dataset/mudah-apartment-kl-selangor_cleaned.csv')

    # Clean dataset columns (normalize column names)
    df.columns = df.columns.str.strip().str.lower()

    # Clean 'monthly_rent' column (remove RM and convert to numeric)
    df['monthly_rent'] = (
        df['monthly_rent']
        .str.replace('RM', '', regex=False)
        .str.replace('per month', '', regex=False)
        .str.replace(' ', '', regex=False)
        .apply(pd.to_numeric, errors='coerce')
    )

    # Drop rows with NaN values in important columns
    df = df.dropna(subset=['monthly_rent', 'prop_name', 'location', 'property_type'])

    return df


df = load_data()

# Geocode and cache locations
@st.cache_data
def geocode_all_locations(locations):
    geolocator = Nominatim(user_agent="my_rent_dashboard (213061@student.upm.edu.my)", timeout=10)
    geocoded_data = {}
    
    for location_name in locations:
        for attempt in range(3):  # Retry up to 3 times
            try:
                location = geolocator.geocode(location_name)
                if location:
                    geocoded_data[location_name] = (location.latitude, location.longitude)
                    break
            except GeocoderTimedOut:
                time.sleep(2)  # Wait before retrying
        else:
            geocoded_data[location_name] = (None, None)
    return geocoded_data

# Cache the geocoded data to a file
@st.cache_data
def cache_geocoded_data():
    locations = df['location'].unique()
    geocoded_data = geocode_all_locations(locations)
    joblib.dump(geocoded_data, "geocoded_data.pkl")
    return geocoded_data

# Load cached geocoded data
@st.cache_data
def load_cached_data():
    try:
        return joblib.load("geocoded_data.pkl")
    except FileNotFoundError:
        return cache_geocoded_data()

geocoded_data = load_cached_data()

# Geocoding function to retrieve latitude and longitude
def geocode_location(location_name):
    return geocoded_data.get(location_name, (None, None))


# Sidebar for user input
st.sidebar.title('Interactive Map for Locations')
st.sidebar.header('Select Location and Property Type')

# Get unique locations and property types from the dataset
locations = df['location'].unique()
property_types = df['property_type'].unique()

# User selects the parameters for filtering data
selected_locations = st.sidebar.multiselect('Select Locations', locations)
selected_property_types = st.sidebar.multiselect('Select Property Types', property_types)

# Filter the dataset based on the selected location and property type
filtered_data = df[
    (df['location'].isin(selected_locations)) &
    (df['property_type'].isin(selected_property_types))
]

# Check if any data is available after filtering
if filtered_data.empty:
    st.warning("No data available for the selected filters.")
else:
    # Initialize map centered at a default location (e.g., Kuala Lumpur)
    m = folium.Map(location=[3.139, 101.6869], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)

    # Iterate through the filtered data to add markers for each property
    for index, row in filtered_data.iterrows():
        prop_name = row['prop_name']
        location_name = row['location']

        # Geocode location name to get latitude and longitude
        lat, lon = geocode_location(location_name)

        if lat and lon:
            folium.Marker(
                location=[lat, lon],
                popup=f"Property: {prop_name}<br>Location: {location_name}<br>Type: {row['property_type']}<br>Price: RM {row['monthly_rent']}",
            ).add_to(marker_cluster)

    # Display the map
    st.write("### Property Locations Map")
    from streamlit_folium import folium_static  # Lazy import for folium in Streamlit
    folium_static(m)
