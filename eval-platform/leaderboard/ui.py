import streamlit as st
import requests
import pandas as pd

API = "http://localhost:8000"

st.title("Panchayat: Eval Leaderboard")

if st.button("Refresh"):
    st.rerun()

# fetch leaderboard
data = requests.get(f"{API}/leaderboard").json()

# show table
df = pd.DataFrame(data)
st.dataframe(df)

# show bar chart

st.bar_chart(df.set_index("model")["avg_ensemble"])