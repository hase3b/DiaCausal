import streamlit as st
from pgmpy.inference import VariableElimination
import networkx as nx
import numpy as np
import pandas as pd

def compute_sensitivity_scores(model, target_variable="Diabetes_binary", perturbation_ratio=0.1):
    infer = VariableElimination(model)
    baseline = infer.query([target_variable], show_progress=False)
    baseline_values = baseline.values
    ancestors = nx.ancestors(model, target_variable)

    scores = []
    for node in ancestors:
        try:
            original_cpd = model.get_cpds(node).copy()
            modified_cpd = original_cpd.copy()
            cpd_values = modified_cpd.values.copy()

            # Get the number of parent configurations (columns)
            num_parent_states = int(np.prod(cpd_values.shape[1:]) if cpd_values.ndim > 1 else 1)
            num_child_states = cpd_values.shape[0]

            # Reshape to (child_states, parent_states)
            cpd_2d = cpd_values.reshape(num_child_states, -1)

            # Perturb the first child state across all parent configurations
            original_probs = cpd_2d[0, :].copy()
            perturbed_probs = np.minimum(original_probs * (1 + perturbation_ratio), 1.0)
            delta = perturbed_probs - original_probs

            # Redistribute the delta proportionally among sibling states
            for parent_state in range(num_parent_states):
                remaining_prob = 1.0 - perturbed_probs[parent_state]
                sibling_probs = cpd_2d[1:, parent_state]
                if sibling_probs.sum() > 0:
                    cpd_2d[1:, parent_state] = sibling_probs * (remaining_prob / sibling_probs.sum())
                else:
                    cpd_2d[1:, parent_state] = remaining_prob / (num_child_states - 1)

            cpd_2d[0, :] = perturbed_probs  # Apply perturbation
            modified_cpd.values = cpd_2d.reshape(modified_cpd.values.shape)

            # Update model and compute sensitivity
            model.remove_cpds(original_cpd)
            model.add_cpds(modified_cpd)
            new_dist = infer.query([target_variable], show_progress=False)
            new_values = new_dist.values

            pct_change = np.sum(np.abs(new_values - baseline_values)) / np.sum(baseline_values) * 100
            scores.append((node, pct_change))

            # Revert changes
            model.remove_cpds(modified_cpd)
            model.add_cpds(original_cpd)
        except Exception as e:
            st.warning(f"Sensitivity analysis failed for {node}: {e}")
            continue

    df = pd.DataFrame(scores, columns=["Node", "Sensitivity (%)"])
    df = df.sort_values("Sensitivity (%)", ascending=False).reset_index(drop=True)
    df.insert(0, "#", range(1, len(df) + 1))

    return df
