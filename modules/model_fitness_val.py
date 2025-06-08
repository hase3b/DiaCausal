import streamlit as st
import hashlib
import json
import pandas as pd
from pgmpy.metrics import structure_score
from modules.bif_loader import load_bif_network
from modules.model_options import model_options
from modules.data_loader import load_data

data = load_data()
data = data.astype(int)
class_var = "Diabetes_binary"

def _hash_model_structure(model):
    """Create a stable hash of model nodes and edges."""
    if model is None:
        return None
    nodes = sorted(model.nodes())
    edges = sorted(model.edges())
    structure_str = json.dumps({"nodes": nodes, "edges": edges})
    return hashlib.md5(structure_str.encode()).hexdigest()

@st.cache_data(show_spinner=True)
def _compute_scores(model_hashes):
    results = []
    for name, model_hash in model_hashes.items():
        _, _, model = load_bif_network(name)  # Already cached
        if model is None:
            continue
        try:
            ll = structure_score(model, data, scoring_method="ll-g")
            bic = structure_score(model, data, scoring_method="bic-d")
            aic = structure_score(model, data, scoring_method="aic-d")
            results.append({
                "Model": name,
                "LogLikelihood": ll,
                "NegBIC": -bic
            })
        except Exception as e:
            st.warning(f"Scoring failed for {name}: {e}")
    return pd.DataFrame(results)

def get_model_scores():
    model_hashes = {}
    for name in model_options:
        _, _, model = load_bif_network(name)  # already cached function
        model_hashes[name] = _hash_model_structure(model)

    return _compute_scores(model_hashes)