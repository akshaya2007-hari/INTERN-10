import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

st.title("Customer Segmentation using Hierarchical Clustering")

# Upload dataset
uploaded_file = st.file_uploader("Upload Customer CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Preprocessing
    df = df.drop("CustomerID", axis=1)

    le = LabelEncoder()
    df["Genre"] = le.fit_transform(df["Genre"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Clustering
    hc = AgglomerativeClustering(n_clusters=5, linkage="ward")
    labels = hc.fit_predict(X_scaled)

    df["Cluster"] = labels

    st.write("Clustered Data:", df.head())

    # Visualization
    fig, ax = plt.subplots()
    ax.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Segmentation")

    st.pyplot(fig)
