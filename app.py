
import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from pygam import LinearGAM, s
import base64

# --- Logo ---
st.set_page_config(page_title="Reach & Frequency Predictor", page_icon="📊")
st.image("logo.png", width=180)

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_excel("CombinedDataV3.xlsx", sheet_name="CombinedData")
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=['Impressions', 'Flight Period', 'Reach', 'Audience Size', 'Frequency', 'Frequency Cap Per Flight'])
    df[['Impressions', 'Audience Size', 'Flight Period', 'Reach', 'Frequency', 'Frequency Cap Per Flight']] =         df[['Impressions', 'Audience Size', 'Flight Period', 'Reach', 'Frequency', 'Frequency Cap Per Flight']].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    df['Log_Impressions'] = np.log1p(df['Impressions'])
    df['Log_Audience'] = np.log1p(df['Audience Size'])
    df['Log_Flight'] = np.log1p(df['Flight Period'])
    df['Log_Reach'] = np.log1p(df['Reach'])
    df['Log_Frequency'] = np.log1p(df['Frequency'])
    df['Log_Frequency Cap Per Flight'] = np.log1p(df['Frequency Cap Per Flight'])
    return df

df = load_data()

X = df[['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight']]
y_reach = df['Log_Reach']
y_frequency = df['Log_Frequency']

# --- Train models ---
@st.cache_resource
def train_models(X, y_reach, y_frequency):
    rf_reach = RandomForestRegressor(n_estimators=500, random_state=42).fit(X, y_reach)
    rf_freq = RandomForestRegressor(n_estimators=500, random_state=42).fit(X, y_frequency)
    gam_reach = LinearGAM(s(0) + s(1) + s(2) + s(3)).fit(X, y_reach)
    gam_freq = LinearGAM(s(0) + s(1) + s(2) + s(3)).fit(X, y_frequency)
    return rf_reach, rf_freq, gam_reach, gam_freq

reach_model_rf, freq_model_rf, gam_reach, gam_freq = train_models(X, y_reach, y_frequency)

# --- Frequency cap conversion ---
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

# --- Prediction logic ---
def predict_metrics(impressions, audience_size, flight_period, frequency_cap):
    log_inputs = np.log1p([impressions, audience_size, flight_period, frequency_cap])
    input_df = pd.DataFrame([log_inputs], columns=['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight'])
    pred = {
        'Reach_RF': np.expm1(reach_model_rf.predict(input_df)[0]),
        'Freq_RF': np.expm1(freq_model_rf.predict(input_df)[0]),
        'Reach_GAM': np.expm1(gam_reach.predict(input_df)[0]),
        'Freq_GAM': np.expm1(gam_freq.predict(input_df)[0])
    }
    pred['Calc_Freq_RF'] = impressions / pred['Reach_RF']
    pred['Calc_Freq_GAM'] = impressions / pred['Reach_GAM']
    return pd.DataFrame([pred])

# --- UI ---
st.title("📊 Reach & Frequency Predictor")
st.markdown("Enter your campaign parameters and compare results from Random Forest and GAM models.")

with st.sidebar:
    st.header("Input Parameters")
    impressions = st.number_input("Impression Volume", min_value=1000, step=1000)
    audience_size = st.number_input("Audience Size", min_value=1000, step=1000)
    flight_period = st.number_input("Flight Period (days)", min_value=1, step=1)
    cap_option = st.selectbox("Frequency Cap Type", ["Day", "Week", "Month", "Life"])
    freq_input = st.number_input(f"Frequency Cap per {cap_option}", min_value=1.0)

frequency_cap = calculate_frequency_cap(freq_input, cap_option, flight_period)
result_df = predict_metrics(impressions, audience_size, flight_period, frequency_cap)

st.subheader("🔍 Model Predictions")
st.dataframe(result_df.T.rename(columns={0: "Value"}))

st.subheader("📈 Reach Comparison Chart")
st.bar_chart(result_df[["Reach_RF", "Reach_GAM"]].T)

# --- CSV Download ---
csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Results as CSV", csv, "model_predictions.csv", "text/csv")
