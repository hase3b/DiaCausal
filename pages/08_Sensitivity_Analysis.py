import streamlit as st
import torch
torch.classes.__path__ = [] # adding this line to manually avoid "examining the path of torch.classes raised"; some streamlit internal error, doesn't affect app.
from modules.model_options import model_options
from modules.sensitivity import compute_sensitivity_scores
from modules.bif_loader import load_bif_network
import logging

# Configure logging to suppress pgmpy INFO messages
logging.getLogger('pgmpy').setLevel(logging.WARNING)

st.set_page_config(page_title="Sensitivity Analysis", page_icon="🔬", layout="wide")


st.title("🔬 Sensitivity Analysis")
st.markdown("Analyze how sensitive the probability of Diabetes is to perturbations in its ancestors’ CPDs (Conditional Probability Distributions) within a selected Bayesian Network.")
st.caption("Note: The following results are based on the preloaded learned structures and expert knowledge graphs. These can be overwritten.")
st.markdown("---")

select_model = st.selectbox("Select Model:", model_options, key="senst_model", index=7)
perturbation_ratio = st.slider("Select Perturbation Ratio:", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
_, _, model = load_bif_network(select_model)

scores_df = compute_sensitivity_scores(model, target_variable="Diabetes_binary", perturbation_ratio=perturbation_ratio)

st.subheader("Sensitivity Output (Cumulative % Change in Marginals of Diabetes)")

if not scores_df.empty:
    with st.container(border=False):
        st.dataframe(
            scores_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "#": st.column_config.NumberColumn(
                    format="%d"
                    )
                    }
                    )
    
st.markdown("---")
