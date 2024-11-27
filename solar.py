import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Title
st.title("Solar Energy Production Predictor")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
temperature = st.sidebar.slider("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.5, value=25.0)
humidity = st.sidebar.slider("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0, value=50.0)
solar_radiation = st.sidebar.slider("Solar Radiation (kW/m²)", min_value=0.0, max_value=2.0, step=0.01, value=1.0)
cloud_cover = st.sidebar.slider("Cloud Cover (%)", min_value=0.0, max_value=100.0, step=1.0, value=20.0)
time_of_day = st.sidebar.slider("Time of Day (hours)", min_value=0, max_value=23, step=1, value=12)

# Input Data
input_data = pd.DataFrame({
    "Temperature": [temperature],
    "Humidity": [humidity],
    "Solar_Radiation": [solar_radiation],
    "Cloud_Cover": [cloud_cover],
    "Time_of_Day": [time_of_day]
})

st.subheader("Input Data")
st.write(input_data)

# Load Dataset
@st.cache
def load_data():
    # Replace this with your dataset file
    url = "https://raw.githubusercontent.com/datasets/solar-power-generation-data/main/solar.csv"
    data = pd.read_csv(url)
    # Replace missing values or clean if necessary
    data.dropna(inplace=True)
    return data

data = load_data()

# Show Dataset
st.subheader("Dataset Preview")
st.write(data.head())

# Feature and Target Split
X = data[["Temperature", "Humidity", "Solar_Radiation", "Cloud_Cover", "Time_of_Day"]]
y = data["Energy_Produced"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
error = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("Model Performance")
st.write(f"Root Mean Squared Error: {error:.2f}")

# Prediction on Input Data
prediction = model.predict(input_data)
st.subheader("Predicted Energy Output")
st.write(f"{prediction[0]:.2f} kWh")

# Plot Feature Importance
st.subheader("Feature Importance")
importance = model.feature_importances_
features = X.columns
plt.bar(features, importance)
plt.xlabel("Features")
plt.ylabel("Importance")
st.pyplot(plt)
