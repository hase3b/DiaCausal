import streamlit as st
import torch
torch.classes.__path__ = [] # adding this line to manually avoid "examining the path of torch.classes raised"; some streamlit internal error, doesn't affect app.
import logging
from modules.model_options import model_options
from modules.bif_loader import load_bif_network
from modules.intervention import measure_intervention_impact, run_intervention_simulation

# Configure logging to suppress pgmpy INFO messages
logging.getLogger('pgmpy').setLevel(logging.WARNING)

st.set_page_config(page_title="Interventional Analysis", page_icon="⚙️", layout="wide")

st.title("⚙️ Interventional Analysis")
st.markdown("""
            Simulate the effect of interventions on variables using the do-calculus. It isolates a treatment's true effect by mathematically 
            simulating a perfect trial—forcing the treatment (do(X)) while blocking confounders (like genetics or lifestyle)—to reveal causation, 
            not just correlation.
            """)
st.caption("Note: The following results are based on the preloaded learned structures and expert knowledge graphs. These can be overwritten.")
st.markdown("---")


st.subheader("Cumulative Impact of Intervention on Effect Variable Under Selected Model")
st.caption("""
            The following table shows the cumulative impact of interventions on the effect variable under the selected model.
            The cumulative impact is calculated as the sum of the net absolute percentage change in the marginal probabilities of the effect variable
            for all states of each intervention variable. The higher the cumulative impact, the more the intervention variable affects the effect variable.
""")
selected_model = st.selectbox("Select Model:", model_options, key="inter_model", index=7)
_, _, model = load_bif_network(selected_model)
model_variables = model.nodes()
effect_var = st.selectbox("Select Effect Variable:", model_variables, key="effect_var", index=4)

full_df = measure_intervention_impact(model, target_var=effect_var)
     
# Display table in a container
with st.container(border=False):
    st.dataframe(
        full_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(
                width=1,
                format="%d"
            )
        }
        )
    
st.markdown("---")
    
if selected_model != "None":
    st.subheader(f"Simulating Manual Interventions using Model: {selected_model}")
    intervention_var = st.selectbox("Select Intervention Variable:", model_variables, key="inter_var", index=2)
    cpd = model.get_cpds(intervention_var)
    state_names = cpd.state_names[intervention_var]
    intervention_state = st.selectbox(f"Set {intervention_var} to state:", state_names, key="inter_state", index=0)
    target_var = st.selectbox("Select Target Variable:", model_variables, key="target_var", index=4)

    if st.button("Run Intervention Simulation"):
        try:
            run_intervention_simulation(model, intervention_var, intervention_state, target_var)
        except Exception as e:
            st.error(f"Intervention failed: {e}")
st.markdown("---")