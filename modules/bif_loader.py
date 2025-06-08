import os
import streamlit as st
from pathlib import Path
from functools import wraps
from datetime import datetime
import hashlib

from pgmpy.readwrite import BIFReader
from modules.graph_editor import Node, Edge

# ----------------- DEBUGGABLE CACHE DECORATOR -----------------

def bif_cache(func):
    """
    More robust cache decorator with content hashing
    """
    @st.cache_data
    def _cached_func(*args, _file_hash=None, **kwargs):
        result = func(*args, **kwargs)
        return result

    @wraps(func)
    def wrapper(*args, **kwargs):
        bif_path = Path(args[0] if args else kwargs.get('bif_path'))
        
        try:
            # Get both mtime and content hash for more reliable change detection
            stat = bif_path.stat()
            mtime = stat.st_mtime
            size = stat.st_size
            
            # Calculate content hash
            with open(bif_path, 'rb') as f:
                content_hash = hashlib.md5(f.read()).hexdigest()
            
            readable_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Use both mtime and content hash as cache key
            cache_key = f"{mtime}-{content_hash}"
            return _cached_func(*args, _file_hash=cache_key, **kwargs)
            
        except Exception as e:
            return _cached_func(*args, _file_hash="error", **kwargs)

    return wrapper

# ----------------- BIF LOADER FUNCTIONS -----------------

@bif_cache
def _load_bif_with_path(bif_path):
    try:
        reader = BIFReader(bif_path)
        model = reader.get_model()

        nodes = [
            Node(id=node, label=node, shape="box", style="rounded", size=35)
            for node in model.nodes()
        ]
        edges = [
            Edge(source=edge[0], target=edge[1], style="solid", width=2)
            for edge in model.edges()
        ]

        return nodes, edges, model
    except Exception as e:
        raise

def load_bif_network(model_name):
    model_to_bif = {
        "Peter-Clark Algorithm": "pc_network.bif",
        "Greedy Equivalence Search": "ges_network.bif",
        "Hill Climbing Search": "hc_network.bif",
        "Simulated Annealing": "sa_network.bif",
        "Evolutionary Algorithm": "ea_network.bif",
        "Multi-Agent Genetic Algorithm": "maga_network.bif",
        "Particle Swarm Optimization": "pso_network.bif",
        "Averaged Structure": "averaged_network.bif",
        "High Confidence Knowledge Graph": "high_confidence_knowledge_graph.bif",
        "Moderate Confidence Knowledge Graph": "moderate_confidence_knowledge_graph.bif",
        "Low Confidence Knowledge Graph": "low_confidence_knowledge_graph.bif"
    }

    bif_file = os.path.join("networks", "cbns", model_to_bif.get(model_name))
    
    if not os.path.exists(bif_file):
        st.error(f"BIF file not found for {model_name}")
        return None, None, None

    try:
        return _load_bif_with_path(bif_file)
    except Exception as e:
        st.error(f"Error loading BIF file: {str(e)}")
        return None, None, None

def load_all_models():
    """Load all models from BIF files into memory using cached loading."""
    learned_models = {
        "Peter-Clark Algorithm": "pc_network.bif",
        "Greedy Equivalence Search": "ges_network.bif",
        "Hill Climbing Search": "hc_network.bif",
        "Simulated Annealing": "sa_network.bif",
        "Evolutionary Algorithm": "ea_network.bif",
        "Multi-Agent Genetic Algorithm": "maga_network.bif",
        "Particle Swarm Optimization": "pso_network.bif",
        "Averaged Structure": "averaged_network.bif"
    }

    knowledge_graphs = {
        "High Confidence": "high_confidence_knowledge_graph.bif",
        "Moderate Confidence": "moderate_confidence_knowledge_graph.bif",
        "Low Confidence": "low_confidence_knowledge_graph.bif"
    }

    loaded_models = {}
    combined_models = {**learned_models, **knowledge_graphs}
    for name, file in combined_models.items():
        path = os.path.join("networks", "cbns", file)
        try:
            _, _, model = _load_bif_with_path(path)
            loaded_models[name] = model
        except Exception as e:
            st.warning(f"Failed to load model '{name}': {e}")

    return learned_models, knowledge_graphs, loaded_models