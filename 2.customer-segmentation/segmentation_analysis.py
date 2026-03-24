import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Loading the Online Retail dataset... (This might take a minute)")
# Use the exact filename you uploaded
df_raw = pd.read_excel('online-retail.xlsx')

# 1. Data Cleaning & RFM Preprocessing
print("Cleaning data and calculating RFM features...")
# Drop rows without a CustomerID
df_clean = df_raw.dropna(subset=['CustomerID'])

# Remove canceled orders (Quantity < 0)
df_clean = df_clean[df_clean['Quantity'] > 0]

# Calculate total price per line item
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

# Convert InvoiceDate to datetime
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

# Define a "snapshot date" (one day after the last transaction in the dataset)
snapshot_date = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)

# Aggregate transactions to the Customer level (RFM Calculation)
rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
    'InvoiceNo': 'nunique',                                  # Frequency
    'TotalPrice': 'sum'                                      # Monetary
}).reset_index()

# Rename columns for clarity
rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)

# Handle extreme outliers (common in retail data) by capping at the 99th percentile
for col in ['Recency', 'Frequency', 'Monetary']:
    cap = rfm[col].quantile(0.99)
    rfm[col] = np.where(rfm[col] > cap, cap, rfm[col])

# Normalization
print("Normalizing features...")
features = ['Recency', 'Frequency', 'Monetary']
X = rfm[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Elbow Method & Silhouette Score Analysis
print("Analyzing optimal clusters on a sample to save memory...")
# We take a sample of 5000 for silhouette score to prevent memory crashes on large datasets
sample_indices = np.random.choice(X_scaled.shape[0], min(5000, X_scaled.shape[0]), replace=False)
X_sample = X_scaled[sample_indices]

inertia = []
silhouette_scores = {}
k_range = range(2, 8) # Testing 2 to 7 clusters

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled) # Fit on all data for inertia
    inertia.append(kmeans.inertia_)
    
    # Calculate silhouette score only on the sample
    kmeans_sample = KMeans(n_clusters=k, random_state=42, n_init=10)
    sample_labels = kmeans_sample.fit_predict(X_sample)
    silhouette_scores[k] = silhouette_score(X_sample, sample_labels)

optimal_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"Optimal number of clusters selected: {optimal_k}")

# 3. Implementation of Multiple ML Models
print("Applying K-Means, DBSCAN, and Hierarchical clustering...")

# K-Means
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['KMeans_Cluster'] = kmeans_final.fit_predict(X_scaled)

# DBSCAN (Density-Based)
dbscan = DBSCAN(eps=0.5, min_samples=10)
rfm['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Hierarchical (Agglomerative)
hc = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
rfm['Hierarchical_Cluster'] = hc.fit_predict(X_scaled)

# 4. Save artifacts for the Streamlit Dashboard
print("Saving processed data and artifacts...")
rfm.to_csv('segmented_customers.csv', index=False)

artifacts = {
    'inertia': inertia,
    'silhouette': silhouette_scores,
    'k_range': list(k_range),
    'optimal_k': optimal_k
}
with open('clustering_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("Success! Run the Streamlit app to view the dashboard.")