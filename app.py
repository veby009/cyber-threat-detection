import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Cyber Threat Detection", layout="wide")

st.title("🛡️ Cyber Threat Detection Dashboard")
st.markdown("Detect anomalies in network traffic using Isolation Forest")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded")

    if 'label' in df.columns:
        y = df['label']
        X = df.drop('label', axis=1)
    else:
        y = None
        X = df.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    preds = model.fit_predict(X_scaled)

    df['anomaly_score'] = preds
    df['predicted_label'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.subheader("🔍 Anomaly Detection Results")
    st.write(df['predicted_label'].value_counts().rename(index={0: "Normal", 1: "Anomaly"}))

    st.subheader("📉 Anomaly Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="anomaly_score", hue="predicted_label", multiple="stack", ax=ax)
    st.pyplot(fig)

    st.download_button("⬇️ Download results as CSV", data=df.to_csv(index=False), file_name="results.csv")

else:
    st.warning("⚠️ Please upload a CSV file to begin.")
