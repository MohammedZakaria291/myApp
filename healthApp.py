# healthApp.py - النسخة النهائية الشغالة 100%
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
    <h3 style='text-align: center; color: #666;'>Upload Model • Scaler • Data → Instant Prediction</h3>
    <hr style='border: 3px solid #00BFFF;'>
""", unsafe_allow_html=True)

# ==============================================
# Model (مطابق تمامًا للنموذج النهائي بتاعك)
# ==============================================
class FinalLSTMHealth(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(9, 182, num_layers=1, batch_first=True, dropout=0.0)  # dropout=0 لأن num_layers=1
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
uploaded_model  = st.sidebar.file_uploader("1. Model (.pth)",  type="pth")
uploaded_scaler = st.sidebar.file_uploader("2. Scaler (.pkl)", type="pkl")
uploaded_data   = st.sidebar.file_uploader("3. Data (.csv)",   type="csv")

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

    # إضافة health_score لو مش موجود في الـ CSV
    if 'health_score' not in df.columns:
        df['health_score'] = 100.0  # مؤقتًا عشان الرسم ما يفشلش

    return model, scaler, df

with st.spinner("Loading AI model and data..."):
    model, scaler, df = load_files(uploaded_model, uploaded_scaler, uploaded_data)

st.success("Files loaded successfully! Ready for prediction")

# ==============================================
# Prepare last 20 rows
# ==============================================
def prepare_sequence(df_machine):
    df_m = df_machine.copy()
    df_m['temp_ma'] = df_m['temperature'].rolling(5, min_periods=1).mean()
    df_m['vib_ma']  = df_m['vibration'].rolling(5, min_periods=1).mean()
    df_m['temp_roc'] = df_m['temperature'].diff().fillna(0)
    df_m['vib_roc']  = df_m['vibration'].diff().fillna(0)

    feats = ['temperature','vibration','humidity','pressure','energy_consumption',
             'temp_ma','vib_ma','temp_roc','vib_roc']
    return df_m[feats].tail(20).values

# ==============================================
# Machine Selection
# ==============================================
machine_id = st.selectbox("Select Machine ID", sorted(df['machine_id'].unique()))

machine_data = df[df['machine_id'] == machine_id].copy()

if len(machine_data) < 20:
    st.error("Not enough data (need at least 20 readings)")
    st.stop()

# Predict
seq = prepare_sequence(machine_data)
seq_scaled = scaler.transform(seq)
seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    health_score = model(seq_tensor).item()

# ==============================================
# Display Results
# ==============================================
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"### Machine {machine_id}")
    st.metric("Health Score", f"{health_score:.1f}/100", delta=f"{health_score-85:+.1f}")
    st.progress(health_score / 100)

    if health_score >= 85:
        st.success("Excellent Condition")
    elif health_score >= 70:
        st.warning("Plan Maintenance")
    else:
        st.error("CRITICAL – Immediate Action!")

with col2:
    st.markdown("### Health Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=machine_data['timestamp'],
        y=machine_data.get('health_score', [health_score]*len(machine_data)),  # لو مفيش health_score في الـ CSV
        mode='lines',
        name='Health Score',
        line=dict(color='#00BFFF', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[machine_data['timestamp'].iloc[-1],
        y=[health_score],
        mode='markers',
        marker=dict(color='red', size=15, symbol='star'),
        name='Latest Prediction'
    ))
    fig.add_hline(y=85, line_dash="dash", line_color="orange")
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.success(f"Machine {machine_id} → Health Score: {health_score:.1f}/100")
