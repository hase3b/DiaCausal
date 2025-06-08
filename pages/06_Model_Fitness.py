import streamlit as st
import torch
torch.classes.__path__ = [] # adding this line to manually avoid "examining the path of torch.classes raised"; some streamlit internal error, doesn't affect app.
import logging
from modules.model_fitness_val import get_model_scores
import plotly.express as px

# Configure logging to suppress pgmpy INFO messages
logging.getLogger('pgmpy').setLevel(logging.WARNING)

st.set_page_config(page_title="Model Fitness", page_icon="📊", layout="wide")

st.title("📊 Model Fitness")
st.markdown("Assess model fitness (BIC, Log-Likelihood).")
st.caption("Note: The following results are based on the preloaded learned structures and expert knowledge graphs. These can be overwritten.")
st.markdown("---")

st.subheader("Assess Model Fit (BIC, Log-Likelihood)")

# Compute scores
score_df = get_model_scores()

# Plot 1
fig1 = px.scatter(
    score_df,
    x="LogLikelihood",
    y="NegBIC",
    color="Model",
    title="Log-Likelihood Vs. BIC Score",
    labels={"LogLikelihood": "Log-Likelihood", "NegBIC": "-BIC Score"},
    )

fig1.update_layout(
    legend_title="Model",
    margin=dict(t=40, b=10)
    )

st.plotly_chart(fig1, use_container_width=True)
st.caption("A two-dimensional scatter plot on the relationship between log-likelihood and the BIC score.")
st.markdown("--- ")