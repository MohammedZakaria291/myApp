# app.py - Fully Working Streamlit App for Your Final Model
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
st.set_page_config(
    page_title="AI Machine Health Monitor",
    page_icon="robot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <h1 style='text-align: center; color: #00BFFF;'>AI-Powered Machine Health Monitor</h1>
    <h3 style='text-align: center; color: #888;'>Real-time Health Score Prediction (0–100)</h3>
    <hr style='border: 3px solid #00BFFF;'>
""", unsafe_allow_html=True)

# ==============================================
# Your Exact Final Model Architecture
# ==============================================
class FinalLSTMHealth(nn.Module):
    def __init__(self):
        super(FinalLSTMHealth, self).__init__()
        self.lstm = nn.LSTM(
            input_size=9,           # 9 features
            hidden_size=182,
            num_layers=1,
            batch_first=True,
            dropout=0.0             # dropout only applies when num_layers > 1
        )
        self.fc = nn.Sequential(
            nn.Linear(182, 64),
            nn.ReLU(),
            nn.Dropout(0.217),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]) * 100

# ==============================================
# Sidebar - File Uploads
# ==============================================
st.sidebar.header("Upload Your Files")
uploaded_model = st.sidebar.file_uploader("1. Trained Model (.pth)", type="pth")
uploaded_scaler = st.sidebar.file_uploader("2. Scaler (.pkl)", type="pkl")
uploaded_data = st.sidebar.file_uploader("3. Dataset (.csv)", type="csv")

if not (uploaded_model and uploaded_scaler and uploaded_data):
    st.info("👆 Please upload all three files from the sidebar to continue.")
    st.stop()

# ==============================================
# Load Model, Scaler, Data
# ==============================================
@st.cache_resource
def load_resources(model_file, scaler_file, data_file):
    # Save temporarily
    with open("temp_model.pth", "wb") as f:
        f.write(model_file.getbuffer())
    with open("temp_scaler.pkl", "wb") as f:
        f.write(scaler_file.getbuffer())

    # Load model
    model = FinalLSTMHealth()
    model.load_state_dict(torch.load("temp_model.pth", map_location="cpu"))
    model.eval()

    # Load scaler
    with open("temp_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load data
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return model, scaler, df

with st.spinner("Loading model, scaler, and data..."):
    model, scaler, df = load_resources(uploaded_model, uploaded_scaler, uploaded_data)

st.success("All files loaded successfully! Ready for prediction.")

# ==============================================
# Feature Engineering Function
# ==============================================
def prepare_features(df_subset):
    df_m = df_subset.copy()
    df_m['temp_ma'] = df_m['temperature'].rolling(5, min_periods=1).mean()
    df_m['vib_ma'] = df_m['vibration'].rolling(5, min_periods=1).mean()
    df_m['temp_roc'] = df_m['temperature'].diff().fillna(0)
    df_m['vib_roc'] = df_m['vibration'].diff().fillna(0)
    features = [
        'temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption',
        'temp_ma', 'vib_ma', 'temp_roc', 'vib_roc'
    ]
    return df_m[features]

# ==============================================
# Machine Selection
# ==============================================
machine_ids = sorted(df['machine_id'].unique())
selected_machine = st.selectbox("Select Machine ID", machine_ids)

machine_data = df[df['machine_id'] == selected_machine].copy()

if len(machine_data) < 20:
    st.error("Not enough data for this machine (minimum 20 readings required)")
    st.stop()

# ==============================================
# Predict Current Health Score
# ==============================================
latest_20 = machine_data.tail(20)
features_20 = prepare_features(latest_20)
scaled_20 = scaler.transform(features_20.values)
tensor_20 = torch.tensor(scaled_20, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    current_health = model(tensor_20).item()

# ==============================================
# Predict Trend (Last 200 readings)
# ==============================================
N = 200
trend_data = machine_data.tail(N)
trend_scores = []
timestamps = []

if len(trend_data) >= 20:
    for i in range(19, len(trend_data)):
        window = trend_data.iloc[i-19:i+1]
        feats = prepare_features(window)
        scaled = scaler.transform(feats.values)
        tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            score = model(tensor).item()
        trend_scores.append(score)
        timestamps.append(trend_data['timestamp'].iloc[i])

# ==============================================
# Display Results
# ==============================================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"### Machine {selected_machine}")
    st.metric("Current Health Score", f"{current_health:.1f}/100", delta=f"{current_health - 85:+.1f}")

    st.progress(current_health / 100)

    if current_health >= 85:
        st.success("Excellent Condition – No Action Needed")
    elif current_health >= 70:
        st.warning("Warning – Schedule Preventive Maintenance")
    else:
        st.error("CRITICAL – Immediate Maintenance Required!")

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
        name='Latest Prediction'
    ))
    fig.add_hline(y=85, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    fig.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ==============================================
# Footer
# ==============================================
st.markdown("---")
st.caption("AI Model: 1-Layer LSTM (182 units) • R² > 0.91 • Trained on 98k+ sensor readings • Built with Streamlit & PyTorch")
