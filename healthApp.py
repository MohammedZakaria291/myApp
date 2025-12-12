# healthApp.py - شغال 100% على Streamlit Cloud
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="AI Machine Health Monitor", layout="wide", page_icon="robot")

st.markdown("""
    <h1 style='text-align: center; color: #00BFFF;'>AI Machine Health Monitor</h1>
    <h3 style='text-align: center; color: #888;'>Upload Model • Scaler • Data → Instant Prediction</h3>
    <hr style='border: 3px solid #00BFFF;'>
""", unsafe_allow_html=True)

# النموذج الأصلي اللي دربت بيه (مهم جدًا يكون زي كده)
class LSTMHealth(nn.Module):
    def __init__(self):
        super(LSTMHealth, self).__init__()
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

# Sidebar
st.sidebar.header("Upload Files")
uploaded_model = st.sidebar.file_uploader("Model (.pth)", type="pth")
uploaded_scaler = st.sidebar.file_uploader("Scaler (.pkl)", type="pkl")
uploaded_data = st.sidebar.file_uploader("Data (.csv)", type="csv")

if not all([uploaded_model, uploaded_scaler, uploaded_data]):
    st.info("Please upload all three files.")
    st.stop()

# Load
@st.cache_resource
def load_everything(m, s, c):
    with open("m.pth", "wb") as f: f.write(m.getbuffer())
    with open("s.pkl", "wb") as f: f.write(s.getbuffer())

    model = LSTMHealth()
    model.load_state_dict(torch.load("m.pth", map_location='cpu'))
    model.eval()

    with open("s.pkl", "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(c)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return model, scaler, df

with st.spinner("Loading model and data..."):
    model, scaler, df = load_everything(uploaded_model, uploaded_scaler, uploaded_data)
st.success("Loaded successfully!")

# Feature prep
def prepare_seq(df_m):
    df_m = df_m.copy()
    df_m['temp_ma'] = df_m['temperature'].rolling(5, min_periods=1).mean()
    df_m['vib_ma'] = df_m['vibration'].rolling(5, min_periods=1).mean()
    df_m['temp_roc'] = df_m['temperature'].diff().fillna(0)
    df_m['vib_roc'] = df_m['vibration'].diff().fillna(0)
    feats = ['temperature','vibration','humidity','pressure','energy_consumption',
             'temp_ma','vib_ma','temp_roc','vib_roc']
    return df_m[feats].tail(20).values

# Machine selection
machine_id = st.selectbox("Select Machine ID", sorted(df['machine_id'].unique()))
data = df[df['machine_id'] == machine_id]

if len(data) < 20:
    st.error("Not enough data")
    st.stop()

# Predict current health
seq = prepare_seq(data)
scaled = scaler.transform(seq)
tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    health = model(tensor).item()

# Display
col1, col2 = st.columns([1,2])
with col1:
    st.metric("Health Score", f"{health:.1f}/100")
    st.progress(health/100)
    if health >= 85: st.success("Excellent")
    elif health >= 70: st.warning("Plan Maintenance")
    else: st.error("CRITICAL")

with col2:
    st.markdown("### Health Trend (Last 200 Points)")
    recent = data.tail(200)
    seqs = []
    for i in range(19, len(recent)):
        tmp = recent.iloc[i-19:i+1].copy()
        seqs.append(prepare_seq(tmp))
    
    scores = []
    for s in seqs:
        sc = scaler.transform(s)
        t = torch.tensor(sc, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            scores.append(model(t).item())
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=scores, mode='lines+markers', line=dict(color='#00BFFF')))
    fig.add_hline(y=85, line_dash="dash", line_color="orange")
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

st.success(f"Machine {machine_id} → Health: {health:.1f}/100")
