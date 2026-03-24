import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pickle
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(page_title="Global Asset Forecaster", layout="wide")

# Custom CSS for UI styling
st.markdown("""
    <style>
    div.stButton > button {
        border-radius: 0.75rem !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        background-color: #111827;
        color: white;
        transition: all 0.2s;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #374151;
        border-color: #d1d5db !important;
    }
    .panel {
        background-color: #ffffff;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = load_model('best_stock_model.h5')
    with open('stock_scaler.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    return model, artifacts['scaler'], artifacts['seq_length']

try:
    model, scaler, seq_length = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Please run train_stock_model.py first!")
    st.stop()

st.title("📈 Global Asset Price Prediction Engine")
st.markdown("Deep Learning Time-Series Forecasting using LSTM/GRU Networks.")

col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Market Parameters")
    
    # Global assets focus
    asset = st.selectbox("Select Global Asset:", ["BTC-USD", "ETH-USD", "^GSPC", "AAPL", "TSLA"])
    timeframe = st.selectbox("Historical Lookback:", ["1 Year", "2 Years", "5 Years"])
    
    period_map = {"1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
    
    analyze_btn = st.button("Generate Forecast")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if analyze_btn:
        with st.spinner(f"Fetching live market data for {asset} and generating neural network predictions..."):
            # 1. Fetch live data
            data = yf.download(asset, period=period_map[timeframe])
            df = data[['Close']].dropna()
            
            # 2. Preprocess data
            scaled_data = scaler.transform(df)
            
            # 3. Create prediction sequence for historical plotting
            X_viz = []
            for i in range(len(scaled_data) - seq_length):
                X_viz.append(scaled_data[i:(i + seq_length), 0])
            X_viz = np.array(X_viz)
            X_viz = np.reshape(X_viz, (X_viz.shape[0], X_viz.shape[1], 1))
            
            # Predict historical data to show model fit
            predictions_scaled = model.predict(X_viz)
            predictions = scaler.inverse_transform(predictions_scaled)
            
            # Align dates for plotting
            valid_dates = df.index[seq_length:]
            actual_prices = df['Close'].values[seq_length:]
            
            # 4. Predict NEXT day's price
            last_sequence = scaled_data[-seq_length:]
            last_sequence = np.reshape(last_sequence, (1, seq_length, 1))
            next_day_scaled = model.predict(last_sequence)
            next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]
            
            # --- Results UI ---
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.metric(label=f"Predicted Next Day Closing Price ({asset})", value=f"${next_day_price:,.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # --- Visualization ---
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            fig = go.Figure()
            
            # Plot Actual Prices
            fig.add_trace(go.Scatter(x=valid_dates, y=actual_prices.flatten(), mode='lines', name='Actual Price', line=dict(color='#3b82f6', width=2)))
            
            # Plot Predicted Prices
            fig.add_trace(go.Scatter(x=valid_dates, y=predictions.flatten(), mode='lines', name='AI Prediction', line=dict(color='#ef4444', width=2, dash='dot')))
            
            fig.update_layout(
                title=f"{asset} Price Projection Model",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                plot_bgcolor='white',
                hovermode='x unified',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
         st.info("Select an asset and click 'Generate Forecast' to run the deep learning sequence.")