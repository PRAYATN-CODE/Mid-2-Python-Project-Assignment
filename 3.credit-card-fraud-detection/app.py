import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# Set up page configuration for an enterprise look
st.set_page_config(page_title="Global Fraud Intelligence", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS applying your specific design requirements
st.markdown("""
    <style>
    /* Styling buttons */
    div.stButton > button {
        border-radius: 0.75rem !important; /* rounded-xl */
        border: 1px solid #e5e7eb !important; /* border-gray-200 */
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important; /* shadow-sm */
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #1d4ed8;
        border-color: #d1d5db !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    /* Styling the main container boxes */
    .dashboard-card {
        background-color: #ffffff;
        border-radius: 0.75rem; /* rounded-xl */
        border: 1px solid #e5e7eb; /* border-gray-200 */
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-sm */
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    /* Adjusting background color for contrast */
    .stApp {
        background-color: #f9fafb;
    }
    h1, h2, h3, p {
        color: #1f2937;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    with open('fraud_detection_artifacts.pkl', 'rb') as f:
        return pickle.load(f)

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
features = artifacts['features']

st.title("🛡️ Global Fraud Intelligence Engine")
st.markdown("Enter transaction parameters below to simulate real-time ML risk assessment.")

# Layout: Two main columns
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
    st.subheader("Transaction Parameters")
    
    # Logical grouping of inputs
    st.markdown("##### 📍 Location Details")
    dist_home = st.number_input("Distance from Home (Miles)", min_value=0.0, value=5.0, step=1.0)
    dist_last = st.number_input("Distance from Last Transaction", min_value=0.0, value=1.0, step=1.0)
    
    st.markdown("##### 💳 Purchase Details")
    ratio_median = st.slider("Ratio to Median Purchase Price", min_value=0.0, max_value=20.0, value=1.0, step=0.1, 
                             help="E.g., 2.0 means this purchase is twice as large as their usual purchases.")
    
    st.markdown("##### 🔒 Authentication & Behavior")
    # Using columns for binary toggles to save space
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        repeat_retailer = st.selectbox("Repeat Retailer?", ["Yes", "No"])
        used_chip = st.selectbox("Used Physical Chip?", ["Yes", "No"])
    with b_col2:
        used_pin = st.selectbox("Used PIN Number?", ["Yes", "No"])
        online_order = st.selectbox("Online Order?", ["Yes", "No"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Authorize Transaction")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if analyze_btn:
        with st.spinner("Analyzing risk footprint..."):
            # Convert UI inputs to binary format for the model
            input_dict = {
                'distance_from_home': dist_home,
                'distance_from_last_transaction': dist_last,
                'ratio_to_median_purchase_price': ratio_median,
                'repeat_retailer': 1.0 if repeat_retailer == "Yes" else 0.0,
                'used_chip': 1.0 if used_chip == "Yes" else 0.0,
                'used_pin_number': 1.0 if used_pin == "Yes" else 0.0,
                'online_order': 1.0 if online_order == "Yes" else 0.0
            }
            
            input_df = pd.DataFrame([input_dict])
            scaled_input = scaler.transform(input_df)
            
            # Prediction
            fraud_probability = model.predict_proba(scaled_input)[0][1]
            is_fraud = model.predict(scaled_input)[0]
            
            # --- 1. Top Result Card ---
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            if is_fraud == 1:
                st.error("🚨 **TRANSACTION DECLINED**")
                st.metric(label="Calculated Fraud Probability", value=f"{fraud_probability:.2%}", delta="High Risk", delta_color="inverse")
            else:
                st.success("✅ **TRANSACTION APPROVED**")
                st.metric(label="Calculated Fraud Probability", value=f"{fraud_probability:.2%}", delta="Low Risk", delta_color="normal")
            st.markdown("</div>", unsafe_allow_html=True)

            # --- 2. Explainability Card (SHAP) ---
            st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
            st.subheader("Model Explainability (SHAP)")
            st.write("Visualizing the exact factors driving this decision. Red pushes toward Fraud, Blue pushes toward Normal.")
            
            # SHAP computation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(scaled_input)
            
            # Handle different SHAP version outputs
            if isinstance(shap_values, list):
                fraud_shap_values = shap_values[1]
            else:
                fraud_shap_values = shap_values
                
            fig, ax = plt.subplots(figsize=(10, 5))
            # Create a clean, modern SHAP bar plot
            shap.summary_plot(fraud_shap_values, input_df, plot_type="bar", show=False, color="#ef4444")
            
            # Make the matplotlib background transparent to match the sleek UI
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            
            # Render plot
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Default state before button is clicked
        st.markdown("""
        <div class='dashboard-card' style='text-align: center; padding: 4rem 2rem;'>
            <h3 style='color: #6b7280;'>Awaiting Transaction Data</h3>
            <p style='color: #9ca3af;'>Configure the transaction parameters on the left and execute the model to view the risk assessment and SHAP explainability.</p>
        </div>
        """, unsafe_allow_html=True)