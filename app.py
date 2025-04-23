import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pygam import LinearGAM, s
import math

# --- Page Config ---
st.set_page_config(page_title="Reach & Frequency Predictor", layout="centered")

# --- Load and preprocess data ---
@st.cache_data
def load_and_prepare_data():
    # Load data from the Excel file
    df = pd.read_excel("CombinedDataV3.xlsx", sheet_name="CombinedData")
    
    # Clean up column names and drop missing rows
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=['Impressions', 'Flight Period', 'Reach', 'Audience Size', 'Frequency', 'Frequency Cap Per Flight'])

    # Convert columns to numeric
    df[['Impressions', 'Audience Size', 'Flight Period', 'Reach', 'Frequency', 'Frequency Cap Per Flight']] = df[['Impressions', 'Audience Size', 'Flight Period', 'Reach', 'Frequency', 'Frequency Cap Per Flight']].apply(pd.to_numeric, errors='coerce')

    # Drop rows with any remaining NaN values
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

# --- Train Models ---
@st.cache_resource
def train_models(df):
    X = df[['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight']]
    y_reach = df['Log_Reach']
    y_frequency = df['Log_Frequency']

    # Random Forest Models
    rf_reach = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_reach.fit(X, y_reach)

    rf_freq = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_freq.fit(X, y_frequency)

    # GAM Models
    gam_reach = LinearGAM(s(0) + s(1) + s(2) + s(3)).fit(X, y_reach)
    gam_freq = LinearGAM(s(0) + s(1) + s(2) + s(3)).fit(X, y_frequency)

    return rf_reach, rf_freq, gam_reach, gam_freq

# Train the models
rf_reach, rf_freq, gam_reach, gam_freq = train_models(df)

# --- Utility Functions ---
def calculate_frequency_cap(frequency_input, option, flight_period):
    """Calculate the total frequency cap based on the chosen option."""
    if option == "Day":
        return frequency_input * flight_period
    elif option == "Week":
        return math.ceil(flight_period / 7) * frequency_input
    elif option == "Month":
        return (flight_period / 30) * frequency_input
    elif option == "Life":
        return frequency_input
    else:
        raise ValueError("Invalid frequency cap option.")

def predict_metrics(impressions, audience_size, flight_period, frequency_cap):
    """Make predictions using the trained models."""
    # Log-transform inputs
    log_input = pd.DataFrame([[np.log1p(impressions), np.log1p(audience_size), np.log1p(flight_period), np.log1p(frequency_cap)]],
                             columns=['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight'])

    # Get predictions from both models
    log_rf_reach = rf_reach.predict(log_input)[0]
    log_rf_freq = rf_freq.predict(log_input)[0]
    log_gam_reach = gam_reach.predict(log_input)[0]
    log_gam_freq = gam_freq.predict(log_input)[0]

    # Convert predictions back from log space
    pred_rf_reach = np.expm1(log_rf_reach)
    pred_rf_freq = np.expm1(log_rf_freq)
    pred_gam_reach = np.expm1(log_gam_reach)
    pred_gam_freq = np.expm1(log_gam_freq)

    return pred_rf_reach, pred_rf_freq, pred_gam_reach, pred_gam_freq

# --- App UI ---
st.title("📊 Reach & Frequency Predictor")

# --- User Inputs ---
st.markdown("## 📥 Enter Campaign Details")

# Inputs for the campaign
impressions = st.number_input("Impressions", min_value=1000, max_value=1_000_000_000, step=1000, value=100000)
audience_size = st.number_input("Audience Size", min_value=1000, max_value=1_000_000_000, step=1000, value=250000)
flight_period = st.number_input("Flight Period (in days)", min_value=1, max_value=365, value=30)

# Frequency cap input
freq_cap_input = st.number_input("Frequency Cap Value", min_value=1.0, max_value=50.0, step=0.1, value=3.0)
freq_cap_unit = st.selectbox("Frequency Cap Unit", ["Day", "Week", "Month", "Life"], index=1)

submitted = st.button("🔮 Predict")

if submitted:
    # Calculate frequency cap
    freq_cap = calculate_frequency_cap(freq_cap_input, freq_cap_unit, flight_period)

    # Get predictions from the models
    pred_rf_reach, pred_rf_freq, pred_gam_reach, pred_gam_freq = predict_metrics(impressions, audience_size, flight_period, freq_cap)

    # Calculated frequency (Impressions / Reach)
    calc_rf_freq = impressions / pred_rf_reach if pred_rf_reach != 0 else 0
    calc_gam_freq = impressions / pred_gam_reach if pred_gam_reach != 0 else 0

    # Display the results
    st.subheader("🔮 Predictions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌲 Random Forest")
        st.metric("Predicted Reach", f"{pred_rf_reach:,.0f}")
        st.metric("Predicted Frequency", f"{pred_rf_freq:.2f}")
        st.metric("Calculated Frequency", f"{calc_rf_freq:.2f}")

    with col2:
        st.markdown("### 🌀 GAM Model")
        st.metric("Predicted Reach", f"{pred_gam_reach:,.0f}")
        st.metric("Predicted Frequency", f"{pred_gam_freq:.2f}")
        st.metric("Calculated Frequency", f"{calc_gam_freq:.2f}")

    # Optional debug info
    with st.expander("🐞 Debug Info"):
        st.write("Raw Inputs:")
        st.json({
            "Impressions": impressions,
            "Audience Size": audience_size,
            "Flight Period": flight_period,
            "Frequency Cap": freq_cap,
            "Cap Unit": freq_cap_unit
        })

        st.write("Log Inputs to Model:")
        st.dataframe(pd.DataFrame([[np.log1p(impressions), np.log1p(audience_size), np.log1p(flight_period), np.log1p(freq_cap)]],
                                  columns=['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight']))

        st.write("Raw Model Outputs (log-space):")
        st.json({
            "RF Reach (log)": float(pred_rf_reach),
            "RF Freq (log)": float(pred_rf_freq),
            "GAM Reach (log)": float(pred_gam_reach),
            "GAM Freq (log)": float(pred_gam_freq)
        })
