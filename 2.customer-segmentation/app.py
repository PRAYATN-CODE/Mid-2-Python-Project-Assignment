import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Set up page configuration
st.set_page_config(page_title="Global Retail Segmentation", layout="wide")

st.markdown("""
    <style>
    .metric-container {
        border-radius: 0.75rem; /* rounded-xl */
        border: 1px solid #e5e7eb; /* border-gray-200 */
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-sm */
        padding: 1rem;
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv('segmented_customers.csv')
    with open('clustering_artifacts.pkl', 'rb') as f:
        artifacts_data = pickle.load(f)
    return data, artifacts_data

# Load data safely
df, artifacts = load_data()

st.title("🛍️ Global Retail Customer Segmentation (RFM)")
st.markdown("Analyze customer purchasing behavior using Recency, Frequency, and Monetary metrics.")

# Sidebar Configuration
st.sidebar.header("Dashboard Controls")
model_choice = st.sidebar.selectbox(
    "Select Clustering Algorithm:",
    ("K-Means", "Hierarchical", "DBSCAN")
)

# Determine which column to look at based on the dropdown
cluster_column = f"{model_choice.replace('-', '')}_Cluster"

# --- ROW 1: Metrics and Evaluation Visuals ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Model Evaluation: Elbow Method")
    fig_elbow = px.line(
        x=artifacts['k_range'], y=artifacts['inertia'], 
        markers=True, title="Inertia vs. Number of Clusters",
        labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'}
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

with col2:
    st.markdown("### Model Evaluation: Silhouette Score")
    fig_sil = px.bar(
        x=list(artifacts['silhouette'].keys()), 
        y=list(artifacts['silhouette'].values()),
        title="Silhouette Score per 'k'",
        labels={'x': 'Number of Clusters (k)', 'y': 'Silhouette Score'},
        color_discrete_sequence=['#3b82f6']
    )
    st.plotly_chart(fig_sil, use_container_width=True)

st.markdown("---")

# --- ROW 2: 3D Cluster Visualization ---
st.markdown(f"### 3D Cluster Visualization ({model_choice})")

# Safely cast the selected cluster to a string for coloring
df['Cluster_Label'] = df[cluster_column].astype(str)

# Add a diagnostic check to PROVE the data is changing
with st.expander("🔍 Click here to view raw cluster distribution (Sanity Check)"):
    st.write(f"Count of customers in each {model_choice} cluster:")
    st.write(df[cluster_column].value_counts())

fig_3d = px.scatter_3d(
    df, x='Recency', y='Frequency', z='Monetary',
    color='Cluster_Label',
    title=f"Customer Segments using {model_choice} (RFM)",
    labels={
        'Recency': 'Days Since Last Purchase',
        'Frequency': 'Number of Invoices',
        'Monetary': 'Total Spend (USD)'
    },
    opacity=0.7,
    size_max=8
)

# The 'key' argument forces Streamlit to completely redraw the chart on change!
st.plotly_chart(fig_3d, use_container_width=True, key=f"3d_chart_{model_choice}")


# --- ROW 3: Business Insights ---
st.markdown("---")
st.header("💡 Global Strategy & Insights")

st.write(f"**Live Analysis based on {model_choice} model:**")

# Calculate means for each cluster to derive insights
summary = df.groupby(cluster_column)[['Recency', 'Frequency', 'Monetary']].mean().reset_index()

# Count customers in each cluster
summary['Customer_Count'] = df.groupby(cluster_column)['CustomerID'].count().values

st.dataframe(summary.style.format({
    'Recency': '{:.1f} days', 
    'Frequency': '{:.1f} orders', 
    'Monetary': '${:,.2f}',
    'Customer_Count': '{:,}'
}), use_container_width=True)

st.markdown("""
**Understanding the RFM Segments:**
* **Low Recency, High Frequency, High Monetary:** *Global Champions.* Your best customers. Reward them.
* **High Recency, Low Frequency, Low Monetary:** *Lost/Churned.* Attempt reactivation with aggressive discount campaigns.
* **Low Recency, Low Frequency:** *New Customers.* Target them with onboarding emails to build loyalty.
""")