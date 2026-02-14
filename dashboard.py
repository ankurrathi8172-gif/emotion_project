import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Emotion Dashboard", layout="centered")

st.title("Facial Emotion Recognition Dashboard")
st.write("This dashboard shows detected emotions and report analytics.")

LOG_FILE = "emotion_log.csv"

if not os.path.exists(LOG_FILE):
    st.warning("No emotion log found. Please run main.py first.")

else:
    df = pd.read_csv(LOG_FILE)

    if st.button("Clear Emotion Log"):
        df = df.iloc[0:0]
        df.to_csv(LOG_FILE, index=False)
        st.success("Emotion log cleared. Refresh the page.")

    st.subheader("Emotion Report Table")
    st.dataframe(df, use_container_width=True)

    if len(df) > 0:
        emotion_counts = df["Emotion"].value_counts()

        st.success(f"Most Frequent Emotion: {emotion_counts.idxmax()}")
        st.info(f"Total Records: {len(df)}")

        st.subheader("Bar Graph (Emotion Count)")
        fig, ax = plt.subplots()
        emotion_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.subheader("Pie Chart (Emotion Distribution)")
        fig2, ax2 = plt.subplots()
        emotion_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax2)
        ax2.set_ylabel("")
        st.pyplot(fig2)

        st.download_button(
            label="Download Report CSV",
            data=df.to_csv(index=False),
            file_name="emotion_report.csv",
            mime="text/csv"
        )
