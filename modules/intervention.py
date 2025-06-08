import streamlit as st
import torch
torch.classes.__path__ = [] # adding this line to manually avoid "examining the path of torch.classes raised"; some streamlit internal error, doesn't affect app.
import logging
from pgmpy.inference import VariableElimination
import numpy as np
import pandas as pd

# Configure logging to suppress pgmpy INFO messages
logging.getLogger('pgmpy').setLevel(logging.WARNING)


def measure_intervention_impact(model, target_var = "Diabetes_binary"):
    infer = VariableElimination(model)

    # Get baseline marginal of the target
    original_factor = infer.query([target_var], show_progress=False)
    original_marginal = original_factor.values
    state_names = original_factor.state_names[target_var]

    variable_impact = {}

    for var in model.nodes():
        if var == target_var:
            continue

        cpd = model.get_cpds(var)
        if cpd is None:
            continue
        try:
            states = cpd.state_names[var]
        except:
            continue

        total_change = 0.0

        for state in states:
            try:
                # Apply do-operator
                model_do = model.do([var], inplace=False)
                infer_do = VariableElimination(model_do)
                intervened_factor = infer_do.query(
                    [target_var],
                    evidence={var: state}, 
                    show_progress=False
                    )
                intervened_marginal = intervened_factor.values

                # Compute absolute % change
                pct_change = 100 * np.abs(intervened_marginal - original_marginal) / (original_marginal + 1e-8)
                total_change += np.sum(pct_change)

            except Exception as e:
                st.warning(f"Intervention failed for {var} at state {state}: {e}")
                continue

        variable_impact[var] = total_change

    # Convert to sorted DataFrame
    impact_df = pd.DataFrame(list(variable_impact.items()), columns=["Intervention Variable", f"Cumulative Impact on {target_var} Variable (% Change)"])
    impact_df = impact_df.sort_values(f"Cumulative Impact on {target_var} Variable (% Change)", ascending=False).reset_index(drop=True)

    # Add "#" column starting from 1
    impact_df.insert(0, "#", impact_df.index + 1)

    return impact_df


def run_intervention_simulation(model, intervention_var, intervention_state, target_var):
    infer = VariableElimination(model)

    # Step 1: Original Marginal
    original_marginal = infer.query([target_var], show_progress=False)
    st.markdown(f"### 🔹 Marginal Probability of {target_var} (Before Intervention)")
    st.write(original_marginal)

    # Step 2: Apply do-operator
    model_do = model.do([intervention_var])
    infer_do = VariableElimination(model_do)

    # Step 3: Intervened marginal of target_var
    intervened_marginal = infer_do.query(
         [target_var],
         evidence={intervention_var: intervention_state},
         show_progress=False
         )
    st.markdown(f"### 🔹 Marginal Probability of {target_var} (After Intervention)")
    st.write(intervened_marginal)

    # Step 4: Compute percentage change
    st.markdown(f"### ➕➖ Percentage Change in Marginal Probabilities of {target_var}")
    original_vals = original_marginal.values
    new_vals = intervened_marginal.values
    pct_change = 100 * (new_vals - original_vals) / original_vals

    for i, state in enumerate(original_marginal.state_names[target_var]):
        st.write(f"**State `{state}`**: {pct_change[i]:.2f}% change")