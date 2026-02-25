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

st.title("🛡️ AI Intrusion Detection - SOC Dashboard")
st.markdown("Real-time network security monitoring and threat analysis")

# ---------------------------------------------------
# Prediction Logic
# ---------------------------------------------------
def predict_action(row):
    anomaly_score = row[0]

    if anomaly_score > 0.75:
        return 3   # Isolate (Critical)
    elif anomaly_score > 0.6:
        return 2   # Block (High)
    elif anomaly_score > 0.4:
        return 1   # Alert (Medium)
    else:
        return 0   # Allow (Normal)

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

        # Use first 4 columns
        data = df.iloc[:, :4].values

        actions = [predict_action(row) for row in data]

        df["Action"] = actions
        df["Decision"] = df["Action"].apply(action_name)
        df["Severity"] = df["Action"].apply(severity_level)

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
        # Confusion Matrix (if label exists)
        # ---------------------------------------------------
        if accuracy:
            st.subheader("Confusion Matrix")

            tp = sum((df["Actual"]==1)&(df["Predicted"]==1))
            tn = sum((df["Actual"]==0)&(df["Predicted"]==0))
            fp = sum((df["Actual"]==0)&(df["Predicted"]==1))
            fn = sum((df["Actual"]==1)&(df["Predicted"]==0))

            cm = np.array([[tn,fp],[fn,tp]])

            fig4,ax4 = plt.subplots()
            ax4.imshow(cm)
            ax4.set_xticks([0,1])
            ax4.set_yticks([0,1])
            ax4.set_xticklabels(["Normal","Attack"])
            ax4.set_yticklabels(["Normal","Attack"])

            for i in range(2):
                for j in range(2):
                    ax4.text(j,i,cm[i,j],ha="center",va="center")

            st.pyplot(fig4)

        # ---------------------------------------------------
        # Detailed Table
        # ---------------------------------------------------
        st.subheader("Detailed Traffic Decisions")
        st.dataframe(df)

    else:
        st.info("Upload a dataset to begin analysis")

# ===================================================
# TAB 2 : REAL-TIME MONITORING
# ===================================================
with tab2:

    st.subheader("Live Network Simulation")

    run = st.checkbox("Start Monitoring")

    if "live" not in st.session_state:
        st.session_state.live = pd.DataFrame(
            columns=["Time","anomaly_score","packet_rate",
                     "failed_logins","cpu_usage",
                     "Action","Decision","Severity"]
        )

    placeholder = st.empty()

    while run:
        row = np.random.rand(4)

        action = predict_action(row)

        new_entry = {
            "Time": datetime.now().strftime("%H:%M:%S"),
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

            st.subheader("Live Traffic (Last 20)")
            st.dataframe(df_live.tail(20))

            st.subheader("Live Severity Trend")
            st.line_chart(df_live["Action"])

            if action >=2:
                st.error(f"⚠️ {severity_level(action)} Threat Detected!")

        time.sleep(1)

    if not run:
        st.info("Enable monitoring to simulate real-time traffic")