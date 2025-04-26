import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pygam import LinearGAM, s
import math

st.set_page_config(page_title="Reach & Frequency Predictor", layout="centered")

@st.cache_resource(show_spinner=False)
def load_and_train_models():
    df = pd.read_excel("CombinedDataV3.xlsx", sheet_name="CombinedData")
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=['Impressions', 'Flight Period', 'Reach', 'Audience Size', 'Frequency', 'Frequency Cap Per Flight'])

    df[['Impressions', 'Audience Size', 'Flight Period', 'Reach', 'Frequency', 'Frequency Cap Per Flight']] = \
        df[['Impressions', 'Audience Size', 'Flight Period', 'Reach', 'Frequency', 'Frequency Cap Per Flight']].apply(pd.to_numeric, errors='coerce')

    df.dropna(inplace=True)

    df['Log_Impressions'] = np.log1p(df['Impressions'])
    df['Log_Audience'] = np.log1p(df['Audience Size'])
    df['Log_Flight'] = np.log1p(df['Flight Period'])
    df['Log_Reach'] = np.log1p(df['Reach'])
    df['Log_Frequency'] = np.log1p(df['Frequency'])
    df['Log_Frequency Cap Per Flight'] = np.log1p(df['Frequency Cap Per Flight'])

    X = df[['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight']]
    y_reach = df['Log_Reach']
    y_frequency = df['Log_Frequency']

    reach_model_rf = RandomForestRegressor(n_estimators=500, random_state=42)
    reach_model_rf.fit(X, y_reach)

    freq_model_rf = RandomForestRegressor(n_estimators=500, random_state=42)
    freq_model_rf.fit(X, y_frequency)

    gam_reach = LinearGAM(s(0) + s(1) + s(2) + s(3), verbose=False).fit(X, y_reach)
    gam_freq = LinearGAM(s(0) + s(1) + s(2) + s(3), verbose=False).fit(X, y_frequency)

    return reach_model_rf, freq_model_rf, gam_reach, gam_freq

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
        raise ValueError("Invalid option for frequency cap input.")

def predict_metrics(impressions, audience_size, flight_period, frequency_cap):
    log_impressions = np.log1p(impressions)
    log_audience = np.log1p(audience_size)
    log_flight = np.log1p(flight_period)
    log_frequency_cap = np.log1p(frequency_cap)

    input_data = pd.DataFrame(
        [[log_impressions, log_audience, log_flight, log_frequency_cap]],
        columns=['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight']
    )

    reach_model_rf, freq_model_rf, gam_reach, gam_freq = load_and_train_models()

    log_predicted_reach_rf = reach_model_rf.predict(input_data)[0]
    log_predicted_freq_rf = freq_model_rf.predict(input_data)[0]

    log_predicted_reach_gam = gam_reach.predict(input_data)[0]
    log_predicted_freq_gam = gam_freq.predict(input_data)[0]

    predicted_reach_rf = np.expm1(log_predicted_reach_rf)
    predicted_frequency_rf = np.expm1(log_predicted_freq_rf)

    predicted_reach_gam = np.expm1(log_predicted_reach_gam)
    predicted_frequency_gam = np.expm1(log_predicted_freq_gam)

    return predicted_reach_rf, predicted_frequency_rf, predicted_reach_gam, predicted_frequency_gam

# --- Streamlit UI ---
st.title("Reach & Frequency Predictor")

impressions = st.number_input("Impression Volume", min_value=0, value=5000000)
audience_size = st.number_input("Audience Size", min_value=0, value=1000000)
flight_period = st.number_input("Flight Period (days)", min_value=1, value=30)
frequency_input = st.number_input("Frequency Cap per Unit", min_value=1.0, value=3.0)
option = st.selectbox("Frequency Cap Unit", ["Day", "Week", "Month", "Life"], index=0)

if st.button("Predict"):
    frequency_cap = calculate_frequency_cap(frequency_input, option, flight_period)

    rf_reach, rf_freq, gam_reach, gam_freq = predict_metrics(
        impressions, audience_size, flight_period, frequency_cap
    )

    st.subheader("Predictions")
    st.markdown(f"**Random Forest:** Reach: `{rf_reach:.2f}`, Frequency: `{rf_freq:.2f}`, Calculated Freq: `{impressions / rf_reach:.2f}`")
    st.markdown(f"**GAM Model:** Reach: `{gam_reach:.2f}`, Frequency: `{gam_freq:.2f}`, Calculated Freq: `{impressions / gam_reach:.2f}`")
