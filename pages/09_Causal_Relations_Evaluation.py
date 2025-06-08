import streamlit as st
import torch
torch.classes.__path__ = [] # adding this line to manually avoid "examining the path of torch.classes raised"; some streamlit internal error, doesn't affect app.
from modules.model_options import model_options
from modules.edge_compare import compare_networks
import logging

# Configure logging to suppress pgmpy INFO messages
logging.getLogger('pgmpy').setLevel(logging.WARNING)

st.set_page_config(page_title="Causal Relations Evaluation", page_icon="🔗", layout="wide")

st.title("🔗 Causal Relations Evaluation")
st.markdown("Qualitative assessment of causal connections by comparing whether the learned model has the same causal links as found in the expert knowledge graph.")
st.caption("Note: The following results are based on the preloaded learned structures and expert knowledge graphs. These can be overwritten.")
st.markdown("---")

learned_network = st.selectbox("Select Learned Network:", model_options[0:8], key="learned_network", index=7)
expert_model = st.selectbox("Select Expert Knowledge Graph:", model_options[8:], key="expert_model", index=0)

links_df, summary_df=compare_networks(learned_network, expert_model)

if not links_df.empty:
    with st.container(border=False):
        st.dataframe(
            links_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "#": st.column_config.NumberColumn(
                    format="%d"
                    )
                    }
                    )

if not summary_df.empty:
    with st.container(border=False):
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "#": st.column_config.NumberColumn(
                    format="%d"
                    )
                    }
                    )
st.markdown("---")