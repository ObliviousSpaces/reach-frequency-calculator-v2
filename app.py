import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from pygam import LinearGAM, s
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from datetime import date

# --- Load and Prepare Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("CombinedData_Final.xlsx", sheet_name="CombinedData")
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=['Impressions', 'Flight Period', 'Reach', 'Audience Size', 'Frequency', 'Frequency Cap Per Flight'])
    df[['Impressions', 'Audience Size', 'Flight Period', 'Reach', 'Frequency', 'Frequency Cap Per Flight']] = df[
        ['Impressions', 'Audience Size', 'Flight Period', 'Reach', 'Frequency', 'Frequency Cap Per Flight']
    ].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
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

    reach_model_rf = RandomForestRegressor(n_estimators=500, random_state=42)
    reach_model_rf.fit(X, y_reach)

    freq_model_rf = RandomForestRegressor(n_estimators=500, random_state=42)
    freq_model_rf.fit(X, y_frequency)

    gam_reach = LinearGAM(s(0) + s(1) + s(2) + s(3)).fit(X, y_reach)
    gam_freq = LinearGAM(s(0) + s(1) + s(2) + s(3)).fit(X, y_frequency)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reach_model_svm = SVR(kernel='rbf', C=10.0, epsilon=0.01)
    reach_model_svm.fit(X_scaled, y_reach)

    freq_model_svm = SVR(kernel='rbf', C=10.0, epsilon=0.01)
    freq_model_svm.fit(X_scaled, y_frequency)

    return reach_model_rf, freq_model_rf, gam_reach, gam_freq, reach_model_svm, freq_model_svm, scaler

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

def predict_metrics(impressions, audience_size, flight_period, frequency_cap, models):
    reach_model_rf, freq_model_rf, gam_reach, gam_freq, reach_model_svm, freq_model_svm, scaler = models

    log_impressions = np.log1p(impressions)
    log_audience = np.log1p(audience_size)
    log_flight = np.log1p(flight_period)
    log_frequency_cap = np.log1p(frequency_cap)

    input_data = pd.DataFrame(
        [[log_impressions, log_audience, log_flight, log_frequency_cap]],
        columns=['Log_Impressions', 'Log_Audience', 'Log_Flight', 'Log_Frequency Cap Per Flight']
    )

    input_scaled = scaler.transform(input_data)

    log_predicted_reach_rf = reach_model_rf.predict(input_data)[0]
    log_predicted_freq_rf = freq_model_rf.predict(input_data)[0]
    log_predicted_reach_gam = gam_reach.predict(input_data)[0]
    log_predicted_freq_gam = gam_freq.predict(input_data)[0]
    log_predicted_reach_svm = reach_model_svm.predict(input_scaled)[0]
    log_predicted_freq_svm = freq_model_svm.predict(input_scaled)[0]

    predicted_reach_rf = np.expm1(log_predicted_reach_rf)
    predicted_frequency_rf = np.expm1(log_predicted_freq_rf)
    predicted_reach_gam = np.expm1(log_predicted_reach_gam)
    predicted_frequency_gam = np.expm1(log_predicted_freq_gam)
    predicted_reach_svm = np.expm1(log_predicted_reach_svm)
    predicted_frequency_svm = np.expm1(log_predicted_freq_svm)

    return predicted_reach_rf, predicted_frequency_rf, predicted_reach_gam, predicted_frequency_gam, predicted_reach_svm, predicted_frequency_svm

# --- Streamlit App ---
st.set_page_config(page_title="Reach & Frequency Predictor", layout="centered")
st.title("📈 Reach & Frequency Predictor")

with st.spinner('Loading data and training models...'):
    df = load_data()
    models = train_models(df)
st.success('Models ready!')

# --- Set Defaults ---
today = date.today()
default_end = today + pd.Timedelta(days=29)

# --- Initialize session state defaults ---
if "Impression Volume" not in st.session_state:
    st.session_state["Impression Volume"] = 5_000_000
if "Audience Size" not in st.session_state:
    st.session_state["Audience Size"] = 1_000_000
if "Frequency Cap Type" not in st.session_state:
    st.session_state["Frequency Cap Type"] = "Day"
if "Frequency Cap Value" not in st.session_state:
    st.session_state["Frequency Cap Value"] = 3.0
if "Start Date" not in st.session_state:
    st.session_state["Start Date"] = today
if "End Date" not in st.session_state:
    st.session_state["End Date"] = default_end

# --- Reset Button ---
if st.button("🔁 Reset Inputs"):
    st.session_state["Impression Volume"] = 5_000_000
    st.session_state["Audience Size"] = 1_000_000
    st.session_state["Frequency Cap Type"] = "Day"
    st.session_state["Frequency Cap Value"] = 3.0
    st.session_state["Start Date"] = today
    st.session_state["End Date"] = default_end
    st.experimental_rerun()

# --- Campaign Inputs ---
st.header("📊 Campaign Inputs")
col1, col2 = st.columns(2)

with col1:
    impressions = st.number_input(
        "Impression Volume",
        min_value=0,
        value=st.session_state["Impression Volume"],
        step=10000,
        key="Impression Volume"
    )
with col2:
    audience_size = st.number_input(
        "Audience Size",
        min_value=0,
        value=st.session_state["Audience Size"],
        step=10000,
        key="Audience Size"
    )

st.divider()

# --- Frequency Settings ---
st.header("🎯 Frequency Settings")
col3, col4 = st.columns(2)

with col3:
    freq_cap_type = st.selectbox(
        "Frequency Cap Type",
        ["Day", "Week", "Month", "Life"],
        key="Frequency Cap Type"
    )
with col4:
    frequency_input = st.number_input(
        "Frequency Cap Value",
        min_value=0.0,
        value=st.session_state["Frequency Cap Value"],
        step=0.5,
        key="Frequency Cap Value"
    )

st.divider()

# --- Flight Period ---
st.header("📅 Flight Period")
col5, col6 = st.columns(2)

with col5:
    start_date = st.date_input("Start Date", st.session_state["Start Date"], key="Start Date")
with col6:
    end_date = st.date_input("End Date", st.session_state["End Date"], key="End Date")

if start_date > end_date:
    st.error("Start Date must be before End Date!")
    flight_period_days = 0
else:
    flight_period_days = (end_date - start_date).days + 1
    st.success(f"📅 Campaign Length: {flight_period_days} days")

# --- Prediction Button ---
calculate = st.button("🔮 Calculate Predictions")

# --- Calculate Logic ---
if calculate:
    if impressions > 0 and audience_size > 0 and flight_period_days > 0:
        frequency_cap = calculate_frequency_cap(frequency_input, freq_cap_type, flight_period_days)

        predicted_reach_rf, predicted_frequency_rf, predicted_reach_gam, predicted_frequency_gam, predicted_reach_svm, predicted_frequency_svm = predict_metrics(
            impressions, audience_size, flight_period_days, frequency_cap, models
        )

        calculated_frequency_rf = impressions / predicted_reach_rf if predicted_reach_rf else 0
        calculated_frequency_gam = impressions / predicted_reach_gam if predicted_reach_gam else 0
        calculated_frequency_svm = impressions / predicted_reach_svm if predicted_reach_svm else 0

        st.success("Predictions generated successfully.")

        st.header("📈 Prediction Results")
        tab1, tab2, tab3 = st.tabs(["🌲 Random Forest", "🎯 GAM Model", "🧠 SVM Model"])

        with tab1:
            st.metric("Predicted Reach", f"{predicted_reach_rf:,.2f}")
            st.metric("Predicted Frequency", f"{predicted_frequency_rf:.2f}")
            st.metric("Calculated Frequency", f"{calculated_frequency_rf:.2f}")

        with tab2:
            st.metric("Predicted Reach", f"{predicted_reach_gam:,.2f}")
            st.metric("Predicted Frequency", f"{predicted_frequency_gam:.2f}")
            st.metric("Calculated Frequency", f"{calculated_frequency_gam:.2f}")

        with tab3:
            st.metric("Predicted Reach", f"{predicted_reach_svm:,.2f}")
            st.metric("Predicted Frequency", f"{predicted_frequency_svm:.2f}")
            st.metric("Calculated Frequency", f"{calculated_frequency_svm:.2f}")
    else:
        st.warning("Please make sure all fields are filled correctly!")
