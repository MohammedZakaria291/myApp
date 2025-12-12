# healthApp.py - الرسم البياني يعرض Health Score حقيقي من الموديل لكل نقطة
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
    <h3 style='text-align: center; color: #666;'>Real-time Health Score Prediction from Model</h3>
    <hr style='border: 3px solid #00BFFF;'>
""", unsafe_allow_html=True)

# ==============================================
# Model
# ==============================================
class FinalLSTMHealth(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(9, 182, num_layers=1, batch_first=True, dropout=0.0)
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
# Sidebar Upload
# ==============================================
st.sidebar.header("Upload Your Files")
uploaded_model  = st.sidebar.file_uploader("1. Model (.pth)", type="pth")
uploaded_scaler = st.sidebar.file_uploader("2. Scaler (.pkl)", type="pkl")
uploaded_data   = st.sidebar.file_uploader("3. Data (.csv)", type="csv")

if not all([uploaded_model, uploaded_scaler, uploaded_data]):
    st.info("Please upload all three files from the sidebar.")
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

    model = FinalLSTMHealth()
    model.load_state_dict(torch.load("temp_model.pth", map_location='cpu'))
    model.eval()

    with open("temp_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return model, scaler, df

with st.spinner("Loading model and data..."):
    model, scaler, df = load_files(uploaded_model, uploaded_scaler, uploaded_data)

st.success("Files loaded successfully! Ready for prediction")

# ==============================================
# Prepare Features for Sequence
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
    st.error("Not enough data (need at least 20 readings)")
    st.stop()

# ==============================================
# Predict Health Score for every possible sequence (sliding window)
# ==============================================
with st.spinner("Calculating health trend using the model..."):
    df_features = prepare_features(machine_data)
    scaled_features = scaler.transform(df_features)

    health_scores = []
    timestamps = machine_data['timestamp'].values[19:]  # تبدأ من الصف 20

    for i in range(19, len(scaled_features)):
        seq = scaled_features[i-19:i+1]  # 20 rows
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            health = model(seq_tensor).item()
        health_scores.append(health)

    # Latest health (last prediction)
    current_health = health_scores[-1]

# ==============================================
# Display Results
# ==============================================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"### Machine {machine_id}")
    st.metric("Current Health Score", f"{current_health:.1f}/100", delta=f"{current_health-85:+.1f}")
    st.progress(current_health / 100)

    if current_health >= 85:
        st.success("Excellent Condition")
    elif current_health >= 70:
        st.warning("Plan Maintenance")
    else:
        st.error("CRITICAL – Immediate Action!")

with col2:
    st.markdown("### Health Trend (Predicted by Model)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=health_scores,
        mode='lines+markers',
        name='Model-Predicted Health',
        line=dict(color='#00BFFF', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[timestamps[-1]],
        y=[current_health],
        mode='markers',
        marker=dict(color='red', size=15, symbol='star'),
        name='Latest Prediction'
    ))
    fig.add_hline(y=85, line_dash="dash", line_color="orange", annotation_text="Warning")
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Critical")
    fig.update_layout(height=500, template="plotly_white", title=f"Health Trend – Machine {machine_id}")
    st.plotly_chart(fig, use_container_width=True)

st.success(f"Machine {machine_id} → Current Health: {current_health:.1f}/100")
st.caption("Health trend is calculated directly from the AI model for every time point")
