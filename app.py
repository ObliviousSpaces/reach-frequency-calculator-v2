import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pygam import LinearGAM, s
import math
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Reach & Frequency Predictor", layout="centered")

# --- Load and preprocess data ---
@st.cache_data
def load_and_prepare_data():
    df = pd.read_excel("CombinedDataV3.xlsx", sheet_name="CombinedData")
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=['Impressions', 'Flight Period', 'Reach', 'Audience Size', 'Frequency', 'Frequency Cap Per Flight'])
    
    df[['Impressions', 'Audience Size', 'Flight Period', 'Reach', 'Frequency', 'Frequency Cap Per Flight']] = df[['Impressions', 'Audience Size', 'Flight Period', 'Reach', 'Frequency', 'Frequency Cap Per Flight']].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    
    # Add log-transformed features
    df['Log_Impressions'] = np.log1p(df['Impressions'])
    df['Log_Audience'] = np.log1p(df['Audience Size'])
    df['Log_Flight'] = np.log1p(df['Flight Period'])
    df['Log_Reach'] = np.log1p(df['Reach'])
    df['Log_Frequency'] = np.log1p(df['Frequency'])
    df['Log_Frequency Cap Per Flight'] = np.log1p(df['Frequency Cap Per Flight'])

    return df

# Load the data
df = load_and_prepare_data()

# --- Model Predictions (with exponentiation to reverse log transformation) ---
def make_predictions(impressions, audience_size, flight_period, frequency_cap, cap_unit):
    # Log-transformed inputs
    log_impressions = np.log1p(impressions)
    log_audience = np.log1p(audience_size)
    log_flight = np.log1p(flight_period)
    log_frequency_cap = np.log1p(frequency_cap)

    # Prepare input features for model
    input_features = np.array([[log_impressions, log_audience, log_flight, log_frequency_cap]])

    # Predict using pre-trained models
    # For now, using random values for model predictions in log-space
    # Replace with actual model inference calls
    rf_reach_log = 7.000064542696617  # Example log-scale output
    rf_freq_log = 1.2086758740647734
    gam_reach_log = 0.09284591928511024
    gam_freq_log = 0.013043390974182882

    # Exponentiate the log predictions to get back to the original scale
    rf_reach = np.exp(rf_reach_log) - 1  # Reverse log1p
    rf_freq = np.exp(rf_freq_log) - 1
    gam_reach = np.exp(gam_reach_log) - 1
    gam_freq = np.exp(gam_freq_log) - 1

    return rf_reach, rf_freq, gam_reach, gam_freq

# --- Streamlit Input Fields ---
st.title("Reach & Frequency Predictor")

# Input fields with default values
impressions = st.number_input("Impressions", min_value=0, value=5000000)  # Default set to 5,000,000
audience_size = st.number_input("Audience Size", min_value=0, value=1000000)  # Default set to 1,000,000
flight_period = st.number_input("Flight Period (days)", min_value=1, value=30)  # Default set to 30
frequency_cap = st.number_input("Frequency Cap", min_value=1, value=3)  # Default set to 3
cap_unit = st.selectbox("Cap Unit", ["Day", "Week", "Month", "Life"], index=0)  # Default set to "Day"

# Predictions on button click
if st.button("Predict"):
    rf_reach, rf_freq, gam_reach, gam_freq = make_predictions(impressions, audience_size, flight_period, frequency_cap, cap_unit)
    
    # Display results
    st.subheader("Predictions (Original Scale):")
    st.write(f"Random Forest - Reach: {rf_reach:.2f} | Frequency: {rf_freq:.2f}")
    st.write(f"Generalized Additive Model - Reach: {gam_reach:.2f} | Frequency: {gam_freq:.2f}")
