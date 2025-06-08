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

            flat_vals = modified_cpd.values.flatten()
            if len(flat_vals) < 2:
                continue
            # Perturb the first state
            idx = 0
            original = flat_vals[idx]
            perturbed = min(original * (1 + perturbation_ratio), 1.0)
            diff = perturbed - original
            flat_vals[idx] = perturbed
            flat_vals[1] = max(flat_vals[1] - diff, 0.0)
            modified_cpd.values = flat_vals.reshape(modified_cpd.values.shape)

            model.remove_cpds(original_cpd)
            model.add_cpds(modified_cpd)

            new_dist = infer.query([target_variable], show_progress=False)
            new_values = new_dist.values

            pct_change = np.sum(np.abs(new_values - baseline_values)) / np.sum(baseline_values) * 100
            scores.append((node, pct_change))

            model.remove_cpds(modified_cpd)
            model.add_cpds(original_cpd)
        except Exception as e:
            st.warning(f"Sensitivity analysis failed for {node}: {e}")
            continue

    df = pd.DataFrame(scores, columns=["Node", "Sensitivity (%)"])
    df = df.sort_values("Sensitivity (%)", ascending=False).reset_index(drop=True)

    # add # to df as column
    df.insert(0, "#", range(1, len(df) + 1))

    return df