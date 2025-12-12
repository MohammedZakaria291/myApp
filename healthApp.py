# healthApp.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle          # ← هنا الحل
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
    <h1 style='text-align: center; color: #00BFFF;'>
        AI Machine Health Monitor
    </h1>
    <h3 style='text-align: center; color: #666;'>
        Upload your model, scaler & data → Get instant health prediction
    </h3>
    <hr style='border: 3px solid #00BFFF;'>
""", unsafe_allow_html=True)

# ==============================================
# Model (يطابق النموذج النهائي بتاعك)
# ==============================================
class FinalLSTMHealth(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(9, 182, num_layers=1, batch_first=True, dropout=0.217)
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
st.sidebar.header("Upload Files")
uploaded_model  = st.sidebar.file_uploader("Model (.pth)",  type="pth")
uploaded_scaler = st.sidebar.file_uploader("Scaler (.pkl)", type="pkl")
uploaded_data   = st.sidebar.file_uploader("Data (.csv)",   type="csv")

if not all([uploaded_model, uploaded_scaler, uploaded_data]):
    st.info("Please upload the three files from the sidebar.")
    st.stop()

# ==============================================
# Load from uploaded files
# ==============================================
@st.cache_resource
def load_everything(m_file, s_file, csv_file):
    # Save temporarily
    with open("temp_model.pth", "wb") as f:
        f.write(m_file.getbuffer())
    with open("temp_scaler.pkl", "wb") as f:
        f.write(s_file.getbuffer())

    # Load model
    model = FinalLSTMHealth()
    model.load_state_dict(torch.load("temp_model.pth", map_location='cpu'))
    model.eval()

    # Load scaler
    with open("temp_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load data
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return model, scaler, df

with st.spinner("Loading model and data..."):
    model, scaler, df = load_everything(uploaded_model, uploaded_scaler, uploaded_data)

st.success("All files loaded successfully!")

# ==============================================
# Prepare last 20 rows with features
# ==============================================
def get_sequence(machine_df):
    df_m = machine_df.copy()
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

machine_data = df[df['machine_id'] == machine_id]
if len(machine_data) < 20:
    st.error("Not enough data (need ≥20 rows)")
    st.stop()

seq = get_sequence(machine_data)
seq_scaled = scaler.transform(seq)
seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

# Predict
with torch.no_grad():
    health = model(seq_tensor).item()

# ==============================================
# Display
# ==============================================
col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Health Score", f"{health:.1f}/100", delta=f"{health-85:+.1f}")
    st.progress(health/100)
    if health >= 85:
        st.success("Excellent")
    elif health >= 70:
        st.warning("Plan Maintenance")
    else:
        st.error("CRITICAL")

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=machine_data['timestamp'], y=machine_data['health_score'],
                              mode='lines', name='Health', line=dict(color='#00BFFF', width=3)))
    fig.add_trace(go.Scatter(x=[machine_data['timestamp'].iloc[-1]], y=[health],
                              mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Prediction'))
    fig.add_hline(y=85, line_dash="dash", line_color="orange")
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.update_layout(height=500, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.success(f"Machine {machine_id} → Health Score: {health:.1f}/100")
