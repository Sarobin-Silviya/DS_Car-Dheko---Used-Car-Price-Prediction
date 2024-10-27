import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestRegressor

# Custom CSS for larger, bold font across the app
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        font-size: 20px;
        font-weight: bold;
    }
    .stButton>button {
        font-size: 22px;
        font-weight: bold;
    }
    .stSelectbox, .stNumberInput {
        font-size: 20px !important;
    }
    .stMarkdown {
        font-size: 22px;
        font-weight: bold;
    }
    .block-container {
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the MinMaxScaler
scaler_filename = 'minmax_scaler.pkl'
with open(scaler_filename, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load frequency mappings
frequency_mapping_filename = 'frequency_mapping.pkl'
with open(frequency_mapping_filename, 'rb') as freq_file:
    frequency_mappings = pickle.load(freq_file)

# Display an image
image = Image.open(r"C:\Users\SAROBIN SILVIYA\Downloads\New folder\carsdekho img.png")  # Use the uploaded file path
st.image(image, caption="Find Your Car's Value!", use_column_width=True)

# App title with an emoji
st.title("🚗 Car Price Prediction")

# Sidebar for inputs with section headers and emojis
st.sidebar.title("🔍 Enter Car Details")

st.sidebar.markdown("## 🚘 **Basic Information**")
manufacturer = st.sidebar.selectbox("🏭 Manufacturer", frequency_mappings['original equipment manufacturer']['original equipment manufacturer'].tolist())
city = st.sidebar.selectbox("🌆 City", frequency_mappings['city']['city'].tolist())
model_name = st.sidebar.selectbox("🚙 Model", frequency_mappings['model']['model'].tolist())
variant_name = st.sidebar.selectbox("🔧 Variant", frequency_mappings['variantName']['variantName'].tolist())

st.sidebar.markdown("## ⚙️ **Specifications**")
number_of_doors = st.sidebar.number_input("🚪 Number of Doors", min_value=2, max_value=5)
number_of_cylinders = st.sidebar.number_input("🔩 Number of Cylinders", min_value=3, max_value=8)
length = st.sidebar.number_input("📏 Length (mm)", min_value=0)
width = st.sidebar.number_input("📏 Width (mm)", min_value=0)
height = st.sidebar.number_input("📏 Height (mm)", min_value=0)
kilometers_driven = st.sidebar.number_input("📍 Kilometers Driven", min_value=0)
mileage = st.sidebar.number_input("⛽ Mileage (Kmpl)", min_value=0)

st.sidebar.markdown("## 🛠️ **Other Features**")
seats = st.sidebar.number_input("💺 Seats", min_value=2, max_value=10)
model_year = st.sidebar.number_input("📅 Model Year", min_value=2000, max_value=2024)
transmission = st.sidebar.selectbox("🔄 Transmission", ['Manual', 'Automatic'])
drive_type = st.sidebar.selectbox("🛣️ Drive Type", frequency_mappings['drive_type']['drive_type'].tolist())
fuel_type = st.sidebar.selectbox("⛽ Fuel Type", frequency_mappings['Fuel Type']['Fuel Type'].tolist())
color = st.sidebar.selectbox("🎨 Color", frequency_mappings['color']['color'].tolist())
tyre_type = st.sidebar.selectbox("🚗 Tyre Type", frequency_mappings['tyre_type']['tyre_type'].tolist())
body_type = st.sidebar.selectbox("🚙 Body Type", frequency_mappings['Body Type']['Body Type'].tolist())
super_charger = st.sidebar.selectbox("⚡ Super Charger", [0, 1])
turbo_charger = st.sidebar.selectbox("💨 Turbo Charger", [0, 1])
owner_no = st.sidebar.number_input("👤 Owner Number", min_value=1)

st.sidebar.markdown("## 🛡️ **Safety Features**")
driver_airbag = st.sidebar.selectbox("🧑‍✈️ Driver Airbag", [0, 1])
passenger_airbag = st.sidebar.selectbox("👥 Passenger Airbag", [0, 1])
power_steering = st.sidebar.selectbox("🛠️ Power Steering", [0, 1])
power_windows = st.sidebar.selectbox("🔲 Power Windows Front", [0, 1])
air_conditioner = st.sidebar.selectbox("❄️ Air Conditioner", [0, 1])
heater = st.sidebar.selectbox("🔥 Heater", [0, 1])
adjustable_head_lights = st.sidebar.selectbox("💡 Adjustable Head Lights", [0, 1])
manually_adjustable_mirror = st.sidebar.selectbox("🔍 Manually Adjustable Mirror", [0, 1])
centeral_locking = st.sidebar.selectbox("🔒 Central Locking", [0, 1])
insurance_type = st.sidebar.selectbox("🛡️ Insurance Type", frequency_mappings['Insurance Type']['Insurance Type'].tolist())

# Predict button
if st.button("📊 Predict Price"):
    # Prepare input features for the model
    input_data = pd.DataFrame({
        'original equipment manufacturer': [manufacturer],
        'city': [city],
        'model': [model_name],
        'variantName': [variant_name],
        'number_of_doors': [number_of_doors],
        'number_of_cylinders': [number_of_cylinders],
        'length(mm)': [length],
        'width(mm)': [width],
        'height(mm)': [height],
        'kilometers_driven': [kilometers_driven],
        'Mileage(Kmpl)': [mileage],
        'Seats': [seats],
        'modelYear': [model_year],
        'ownerNo': [owner_no],
        'Driver Airbag': [driver_airbag],
        'Passenger Airbag': [passenger_airbag],
        'Power Steering': [power_steering],
        'Power Windows Front': [power_windows],
        'Air Conditioner': [air_conditioner],
        'Heater': [heater],
        'Adjustable Head Lights': [adjustable_head_lights],
        'Manually Adjustable Exterior Rear View Mirror': [manually_adjustable_mirror],
        'Centeral Locking': [centeral_locking],
        'transmission': [transmission],
        'drive_type': [drive_type],
        'Fuel Type': [fuel_type],
        'color': [color],
        'tyre_type': [tyre_type],
        'Body Type': [body_type],
        'super_charger': [super_charger],
        'turbo_charger': [turbo_charger],
        'Insurance Type': [insurance_type],
    })

    # Apply frequency encoding
    for feature in frequency_mappings:
        freq_map = frequency_mappings[feature]
        if feature in input_data.columns:
            input_data[f'{feature}_freq'] = input_data[feature].map(dict(zip(freq_map[feature], freq_map[f'{feature}_freq'])))
            input_data.drop(columns=[feature], inplace=True)

    # Normalize numerical features
    numerical_cols = ['length(mm)', 'width(mm)', 'height(mm)', 'kilometers_driven', 'Mileage(Kmpl)']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Ensure expected feature order for the model
    model_features = model.feature_names_in_
    for feature in model_features:
        if feature not in input_data.columns:
            input_data[feature] = 0
    input_data = input_data[model_features]

    # Make prediction
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")
