# healthApp.py - نسخة سريعة جدًا حتى مع آلات طويلة
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import plotly.graph_objects as go

# ==============================================
# Page Config
# ==============================================
st.set_page_config(page_title="AI Machine Health Monitor", layout="wide", page_icon="robot")

st.markdown("""
    <h1 style='text-align: center; color: #00BFFF;'>AI Machine Health Monitor</h1>
    <h3 style='text-align: center; color: #666;'>Real-time Health Score Prediction (0–100)</h3>
    <hr style='border: 3px solid #00BFFF;'>
""", unsafe_allow_html=True)

# ==============================================
# Model
# ==============================================
class LSTMHealth(nn.Module):
    def __init__(self):
        super().__init()
        self.lstm = nn.LSTM(9, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]) * 100

# ==============================================
# Sidebar Upload
# ==============================================
st.sidebar.header("Upload Files")
uploaded_model = st.sidebar.file_uploader("Model (.pth)", type="pth")
uploaded_scaler = st.sidebar.file_uploader("Scaler (.pkl)", type="pkl")
uploaded_data = st.sidebar.file_uploader("Data (.csv)", type="csv")

if not all([uploaded_model, uploaded_scaler, uploaded_data]):
    st.info("Please upload all three files.")
    st.stop()

# ==============================================
# Load Everything
# ==============================================
@st.cache_resource
def load_files(m_file, s_file, csv_file):
    with open("temp_model.pth", "wb") as f:
        f.write(m_file.getbuffer())
    with open("temp_scaler.pkl", "wb") as f:
        f.write(s_file.getbuffer())

    model = LSTMHealth()
    model.load_state_dict(torch.load("temp_model.pth", map_location='cpu'))
    model.eval()

    with open("temp_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return model, scaler, df

with st.spinner("Loading..."):
    model, scaler, df = load_files(uploaded_model, uploaded_scaler, uploaded_data)

st.success("Loaded successfully!")

# ==============================================
# Feature Preparation
# ==============================================
def prepare_features(df_machine):
    df_m = df_machine.copy()
    df_m['temp_ma'] = df_m['temperature'].rolling(5, min_periods=1).mean()
    df_m['vib_ma']  = df_m['vibration'].rolling(5, min_periods=1).mean()
    df_m['temp_roc'] = df_m['temperature'].diff().fillna(0)
    df_m['vib_roc']  = df_m['vibration'].diff().fillna(0)

    feats = ['temperature','vibration','humidity','pressure','energy_consumption',
             'temp_ma','vib_ma','temp_roc','vib_roc']
    return df_m[feats]

# ==============================================
# Machine Selection
# ==============================================
machine_id = st.selectbox("Select Machine ID", sorted(df['machine_id'].unique()))

machine_data = df[df['machine_id'] == machine_id].copy()

if len(machine_data) < 20:
    st.error("Not enough data")
    st.stop()

# ==============================================
# Predict Current Health (Last 20 readings)
# ==============================================
latest_features = prepare_features(machine_data.tail(20))
latest_scaled = scaler.transform(latest_features)
latest_tensor = torch.tensor(latest_scaled, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    current_health = model(latest_tensor).item()

# ==============================================
# Predict Health Trend on Last N readings (N = 200 for speed)
# ==============================================
N = 200  # عدد القراءات للـ Trend (200 سريع وكافي)
trend_data = machine_data.tail(N).copy()

if len(trend_data) >= 20:
    trend_features = prepare_features(trend_data)
    trend_scaled = scaler.transform(trend_features)

    trend_scores = []
    timestamps = trend_data['timestamp'].values[19:]

    for i in range(19, len(trend_scaled)):
        seq = trend_scaled[i-19:i+1]
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            trend_scores.append(model(seq_tensor).item())
else:
    trend_scores = [current_health] * 5
    timestamps = machine_data['timestamp'].tail(5).values

# ==============================================
# Display Results
# ==============================================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"### Machine {machine_id}")
    st.metric("Current Health Score", f"{current_health:.1f}/100")
    st.progress(current_health / 100)

    if current_health >= 85:
        st.success("Excellent")
    elif current_health >= 70:
        st.warning("Plan Maintenance")
    else:
        st.error("CRITICAL")

with col2:
    st.markdown("### Health Trend (Last 200 Readings)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=trend_scores,
        mode='lines+markers',
        name='Health Score',
        line=dict(color='#00BFFF', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[timestamps[-1]],
        y=[current_health],
        mode='markers',
        marker=dict(color='red', size=15, symbol='star'),
        name='Latest'
    ))
    fig.add_hline(y=85, line_dash="dash", line_color="orange")
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.success(f"Machine {machine_id} Health: {current_health:.1f}/100")
st.caption("Health trend calculated directly from the AI model (last 200 readings)")
