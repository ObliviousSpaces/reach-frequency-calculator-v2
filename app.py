import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from pygam import LinearGAM, s

# --- Load and Prepare Data ---
@st.cache_data

def load_data():
    df = pd.read_excel("CombinedData_Final.xlsx", sheet_name="CombinedData")
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

    return df

# --- Train Models ---
@st.cache_resource

def train_models(df):
    X = df[['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight']]
    y_reach = df['Log_Reach']
    y_frequency = df['Log_Frequency']

    reach_model_rf = RandomForestRegressor(n_estimators=500, random_state=42)
    reach_model_rf.fit(X, y_reach)

    freq_model_rf = RandomForestRegressor(n_estimators=500, random_state=42)
    freq_model_rf.fit(X, y_frequency)

    gam_reach = LinearGAM(s(0) + s(1) + s(2) + s(3)).fit(X, y_reach)
    gam_freq = LinearGAM(s(0) + s(1) + s(2) + s(3)).fit(X, y_frequency)

    return reach_model_rf, freq_model_rf, gam_reach, gam_freq

# --- Helper Function ---
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
        raise ValueError("Invalid frequency cap option.")

# --- Prediction Function ---
def predict_metrics(impressions, audience_size, flight_period, frequency_cap, models):
    reach_model_rf, freq_model_rf, gam_reach, gam_freq = models

    log_impressions = np.log1p(impressions)
    log_audience = np.log1p(audience_size)
    log_flight = np.log1p(flight_period)
    log_frequency_cap = np.log1p(frequency_cap)

    # Debugging: Print log-transformed inputs
    st.write("\n\U0001F50E Log-transformed Inputs:")
    st.write(f"log_impressions = {log_impressions}")
    st.write(f"log_audience = {log_audience}")
    st.write(f"log_flight = {log_flight}")
    st.write(f"log_frequency_cap = {log_frequency_cap}")

    input_data = pd.DataFrame(
        [[log_impressions, log_audience, log_flight, log_frequency_cap]],
        columns=['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight']
    )

    # Predictions
    log_predicted_reach_rf = reach_model_rf.predict(input_data)[0]
    log_predicted_freq_rf = freq_model_rf.predict(input_data)[0]

    log_predicted_reach_gam = gam_reach.predict(input_data)[0]
    log_predicted_freq_gam = gam_freq.predict(input_data)[0]

    # Reverse log transform
    predicted_reach_rf = np.expm1(log_predicted_reach_rf)
    predicted_frequency_rf = np.expm1(log_predicted_freq_rf)

    predicted_reach_gam = np.expm1(log_predicted_reach_gam)
    predicted_frequency_gam = np.expm1(log_predicted_freq_gam)

    return predicted_reach_rf, predicted_frequency_rf, predicted_reach_gam, predicted_frequency_gam

# --- Streamlit App ---

st.set_page_config(page_title="Reach & Frequency Predictor", layout="centered")
st.title("📈 Reach & Frequency Predictor")

# Load and train
with st.spinner('Loading data and training models...'):
    df = load_data()
    models = train_models(df)

st.success('Models ready!')

st.header("🔢 Enter Campaign Details")

col1, col2 = st.columns(2)

with col1:
    impressions = st.number_input("Impression Volume", min_value=0, value=5000000, step=10000)
    audience_size = st.number_input("Audience Size", min_value=0, value=1000000, step=10000)

with col2:
    frequency_input = st.number_input("Frequency Cap", min_value=0.0, value=3.0, step=0.5)
    flight_period = st.number_input("Flight Period (days)", min_value=1, value=30)

option = st.selectbox("Frequency Cap Type", options=["Day", "Week", "Month", "Life"])

# Predict button
if st.button("Predict!"):
    frequency_cap = calculate_frequency_cap(frequency_input, option, flight_period)

    predicted_reach_rf, predicted_frequency_rf, predicted_reach_gam, predicted_frequency_gam = \
        predict_metrics(impressions, audience_size, flight_period, frequency_cap, models)

    calculated_frequency_rf = impressions / predicted_reach_rf if predicted_reach_rf else 0
    calculated_frequency_gam = impressions / predicted_reach_gam if predicted_reach_gam else 0

    st.header("🔮 Predictions:")
    col_rf, col_gam = st.columns(2)

    with col_rf:
        st.subheader("Random Forest Model")
        st.metric(label="Predicted Reach", value=f"{predicted_reach_rf:,.2f}")
        st.metric(label="Predicted Frequency", value=f"{predicted_frequency_rf:,.2f}")
        st.metric(label="Calculated Frequency", value=f"{calculated_frequency_rf:,.2f}")

    with col_gam:
        st.subheader("GAM Model")
        st.metric(label="Predicted Reach", value=f"{predicted_reach_gam:,.2f}")
        st.metric(label="Predicted Frequency", value=f"{predicted_frequency_gam:,.2f}")
        st.metric(label="Calculated Frequency", value=f"{calculated_frequency_gam:,.2f}")
