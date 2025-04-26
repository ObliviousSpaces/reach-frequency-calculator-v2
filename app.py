import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pygam import LinearGAM, s
import math

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_excel(r"C:\Users\paris\Downloads\CombinedDataV3.xlsx", sheet_name='CombinedData')
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

@st.cache_resource
def train_models(df):
    X = df[['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight']]
    y_reach = df['Log_Reach']
    y_frequency = df['Log_Frequency']

    rf_reach = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_reach.fit(X, y_reach)

    rf_freq = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_freq.fit(X, y_frequency)

    gam_reach = LinearGAM(s(0) + s(1) + s(2) + s(3))
    gam_reach.fit(X, y_reach)

    gam_freq = LinearGAM(s(0) + s(1) + s(2) + s(3))
    gam_freq.fit(X, y_frequency)

    return rf_reach, rf_freq, gam_reach, gam_freq

# Frequency Cap calculation
def calculate_frequency_cap(frequency_input, option, flight_period):
    if option == "Day":
        return frequency_input * flight_period
    elif option == "Week":
        return math.ceil(flight_period / 7) * frequency_input
    elif option == "Month":
        return (flight_period / 30) * frequency_input
    elif option == "Life":
        return frequency_input
    else:
        raise ValueError("Invalid option.")

# Main prediction function
def predict_metrics(impressions, audience_size, flight_period, frequency_cap, reach_model_rf, freq_model_rf, gam_reach, gam_freq):
    log_impressions = np.log1p(impressions)
    log_audience = np.log1p(audience_size)
    log_flight = np.log1p(flight_period)
    log_frequency_cap = np.log1p(frequency_cap)

    # DEBUG: Print inputs!
    st.write("🔎 **Log-transformed Inputs:**")
    st.write(f"log_impressions = {log_impressions}")
    st.write(f"log_audience = {log_audience}")
    st.write(f"log_flight = {log_flight}")
    st.write(f"log_frequency_cap = {log_frequency_cap}")

    input_data = pd.DataFrame(
        [[log_impressions, log_audience, log_flight, log_frequency_cap]],
        columns=['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight']
    )

    # Predict
    log_pred_reach_rf = reach_model_rf.predict(input_data)[0]
    log_pred_freq_rf = freq_model_rf.predict(input_data)[0]

    log_pred_reach_gam = gam_reach.predict(input_data)[0]
    log_pred_freq_gam = gam_freq.predict(input_data)[0]

    # Reverse log transform
    pred_reach_rf = np.expm1(log_pred_reach_rf)
    pred_freq_rf = np.expm1(log_pred_freq_rf)
    pred_reach_gam = np.expm1(log_pred_reach_gam)
    pred_freq_gam = np.expm1(log_pred_freq_gam)

    return pred_reach_rf, pred_freq_rf, pred_reach_gam, pred_freq_gam

# Streamlit UI
st.title("🔧 Debugging Reach & Frequency Predictions")

df = load_data()
reach_model_rf, freq_model_rf, gam_reach, gam_freq = train_models(df)

# --- User Inputs
user_impressions = st.number_input("Enter Impressions", min_value=1, value=500000)
user_audience_size = st.number_input("Enter Audience Size", min_value=1, value=100000)
user_flight_period = st.number_input("Enter Flight Period (Days)", min_value=1, value=30)

cap_option = st.selectbox("Frequency Cap Type", ["Day", "Week", "Month", "Life"])
cap_value = st.number_input(f"Frequency Cap per {cap_option}", min_value=0.0, value=2.0)

# --- Calculate total frequency cap
freq_cap = calculate_frequency_cap(cap_value, cap_option, user_flight_period)

# --- Predict
if st.button("Predict!"):
    pred_reach_rf, pred_freq_rf, pred_reach_gam, pred_freq_gam = predict_metrics(
        user_impressions, user_audience_size, user_flight_period, freq_cap,
        reach_model_rf, freq_model_rf, gam_reach, gam_freq
    )

    # --- Results
    st.header("🔮 Predictions:")
    st.subheader("Random Forest Model")
    st.write(f"Predicted Reach: {pred_reach_rf:,.2f}")
    st.write(f"Predicted Frequency: {pred_freq_rf:.2f}")
    st.write(f"Calculated Frequency: {user_impressions / pred_reach_rf:.2f}")

    st.subheader("GAM Model")
    st.write(f"Predicted Reach: {pred_reach_gam:,.2f}")
    st.write(f"Predicted Frequency: {pred_freq_gam:.2f}")
    st.write(f"Calculated Frequency: {user_impressions / pred_reach_gam:.2f}")
