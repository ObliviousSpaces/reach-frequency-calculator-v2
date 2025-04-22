import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pygam import LinearGAM, s
from scipy.sparse import issparse

@st.cache_data
def load_data():
    import os
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), "CombinedDataV3.xlsx"), sheet_name='CombinedData')
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(subset=['Impressions', 'Flight Period', 'Reach', 'Audience Size', 'Frequency', 'Frequency Cap Per Flight'])
    cols = ['Impressions', 'Audience Size', 'Flight Period', 'Reach', 'Frequency', 'Frequency Cap Per Flight']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    df['Log_Impressions'] = np.log1p(df['Impressions'])
    return df

st.title("Reach & Frequency Calculator")
st.write("pygam version:", __import__('pygam').__version__)
df = load_data()
st.dataframe(df.head())