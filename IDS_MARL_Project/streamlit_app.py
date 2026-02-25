import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="AI IDS SOC Dashboard", layout="wide")

st.title("🛡️ AI Intrusion Detection - Multi-Cloud SOC Dashboard")
st.markdown("Real-time security monitoring for AWS | Azure | GCP")

# ---------------------------------------------------
# Cloud Mapping (NEW - Multi Cloud)
# ---------------------------------------------------
cloud_map = {
    0: "AWS",
    1: "Azure",
    2: "GCP"
}

# ---------------------------------------------------
# Prediction Logic
# ---------------------------------------------------
def predict_action(row):
    anomaly_score = row[0]

    if anomaly_score > 0.75:
        return 3   # Isolate
    elif anomaly_score > 0.6:
        return 2   # Block
    elif anomaly_score > 0.4:
        return 1   # Alert
    else:
        return 0   # Allow

def action_name(a):
    return {0:"Allow",1:"Alert",2:"Block",3:"Isolate"}[a]

def severity_level(a):
    return {0:"Low",1:"Medium",2:"High",3:"Critical"}[a]

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2 = st.tabs(["📁 Dataset Analysis", "⚡ Real-Time Monitoring"])

# ===================================================
# TAB 1 : DATASET ANALYSIS
# ===================================================
with tab1:

    st.sidebar.header("Upload Network Dataset")
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Use first 4 columns as features
        data = df.iloc[:, :4].values
        actions = [predict_action(row) for row in data]

        df["Action"] = actions
        df["Decision"] = df["Action"].apply(action_name)
        df["Severity"] = df["Action"].apply(severity_level)

        # ---------------------------------------------------
        # Multi-Cloud Column (if exists)
        # ---------------------------------------------------
        if "cloud_id" in df.columns:
            df["Cloud"] = df["cloud_id"].map(cloud_map)

        # Accuracy
        accuracy = None
        if "label" in df.columns:
            df["Actual"] = df["label"].apply(lambda x: 1 if x == 1 else 0)
            df["Predicted"] = df["Action"].apply(lambda x: 1 if x>0 else 0)
            accuracy = (df["Actual"] == df["Predicted"]).mean()*100

        # ---------------------------------------------------
        # Executive Summary
        # ---------------------------------------------------
        st.subheader("📊 Executive Summary")

        total = len(df)
        attacks = sum(df["Action"]>0)
        normal = sum(df["Action"]==0)
        critical = sum(df["Action"]==3)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Traffic", total)
        c2.metric("Normal", normal)
        c3.metric("Threats", attacks)
        c4.metric("Critical Attacks", critical)

        if accuracy:
            st.success(f"Model Accuracy: {accuracy:.2f}%")

        # ---------------------------------------------------
        # Multi-Cloud Distribution (NEW)
        # ---------------------------------------------------
        if "Cloud" in df.columns:
            st.subheader("☁️ Multi-Cloud Traffic Distribution")
            st.bar_chart(df["Cloud"].value_counts())

        # ---------------------------------------------------
        # Threat Distribution
        # ---------------------------------------------------
        st.subheader("Threat Severity Distribution")

        col1,col2 = st.columns(2)

        with col1:
            fig,ax = plt.subplots()
            df["Severity"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

        with col2:
            fig2,ax2 = plt.subplots()
            df["Decision"].value_counts().plot(kind="bar", ax=ax2)
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

        # ---------------------------------------------------
        # Anomaly Score Analysis
        # ---------------------------------------------------
        st.subheader("Anomaly Score Distribution")

        fig3,ax3 = plt.subplots()
        ax3.hist(df.iloc[:,0], bins=30)
        ax3.set_xlabel("Anomaly Score")
        ax3.set_ylabel("Frequency")
        st.pyplot(fig3)

        # ---------------------------------------------------
        # Detailed Table
        # ---------------------------------------------------
        st.subheader("Detailed Traffic Decisions")
        st.dataframe(df)

    else:
        st.info("Upload a dataset to begin analysis")

# ===================================================
# TAB 2 : REAL-TIME MONITORING (Multi-Cloud Simulation)
# ===================================================
with tab2:

    st.subheader("Live Multi-Cloud Network Simulation")

    run = st.checkbox("Start Monitoring")

    if "live" not in st.session_state:
        st.session_state.live = pd.DataFrame(
            columns=[
                "Time","Cloud",
                "anomaly_score","packet_rate",
                "failed_logins","cpu_usage",
                "Action","Decision","Severity"
            ]
        )

    placeholder = st.empty()

    while run:
        # Generate random network state
        row = np.random.rand(4)

        # Random cloud source
        cloud_id = np.random.randint(0,3)
        cloud_name = cloud_map[cloud_id]

        action = predict_action(row)

        new_entry = {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Cloud": cloud_name,
            "anomaly_score": row[0],
            "packet_rate": row[1],
            "failed_logins": row[2],
            "cpu_usage": row[3],
            "Action": action,
            "Decision": action_name(action),
            "Severity": severity_level(action)
        }

        st.session_state.live = pd.concat(
            [st.session_state.live, pd.DataFrame([new_entry])],
            ignore_index=True
        )

        df_live = st.session_state.live

        with placeholder.container():

            total = len(df_live)
            attacks = sum(df_live["Action"]>0)
            critical = sum(df_live["Action"]==3)

            c1,c2,c3 = st.columns(3)
            c1.metric("Total Traffic", total)
            c2.metric("Threats", attacks)
            c3.metric("Critical", critical)

            st.subheader("Cloud Distribution (Live)")
            st.bar_chart(df_live["Cloud"].value_counts())

            st.subheader("Live Traffic (Last 20)")
            st.dataframe(df_live.tail(20))

            st.subheader("Live Threat Trend")
            st.line_chart(df_live["Action"])

            # Alert for high threats
            if action >=2:
                st.error(f"⚠️ {severity_level(action)} Threat Detected in {cloud_name}!")

        time.sleep(1)

    if not run:
        st.info("Enable monitoring to simulate real-time multi-cloud traffic")