import streamlit as st
import sys
import os
from streamlit_agraph import Node # Keep Node for type hinting if needed
import torch
torch.classes.__path__ = [] # adding this line to manually avoid "examining the path of torch.classes raised"; some streamlit internal error, doesn't affect app.

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary functions from the updated graph_editor module
from modules.graph_editor import (
    initialize_graph_state, # Still useful for initial setup
    display_graph_with_manual_controls,
    VARIABLES, # Import variables list
    get_expert_network_files, # Get list of saved networks (standard names)
    save_graph_to_file,       # Save uploaded graph state
    load_graph_from_file,     # Load graph state from file
    parse_graph_data,         # Import the parsing function
    save_defined_network,     # Import the Rule 6 saving logic
    NETWORK_FILENAMES         # Import the standard filenames mapping
)

st.set_page_config(page_title="Clinical Knowledge Graph", page_icon="🧠", layout="wide")

st.title("🧠 Clinical Knowledge Graph (Expert Input)")
st.markdown("""
    Upload, define, or view expert causal graph structures.
    
    A knowledge expert can define a causal graph structure, assigning edges with varying confidence levels. Which eventually renders three 
    causal graphs, one for each confidence level. The expert can also upload them in a csv format as given in reference paper. The report details 
    the reasoning behind the confidence levels.
    """)
st.caption("Note: Currently, the application is preloaded with the expert graph structures for each confidence level given in the reference paper. These can be overwritten.")
st.markdown("---")

# --- Robust State Initialization --- 
# Initialize mode and loaded network name if they don't exist
if "graph_mode" not in st.session_state:
    st.session_state.graph_mode = "none" # Modes: 'none', 'view', 'define'
if "loaded_network_name" not in st.session_state:
    st.session_state.loaded_network_name = None
if "last_selected_action" not in st.session_state:
    st.session_state.last_selected_action = "Select Action..."

# Initialize graph nodes/edges ONLY if they don't exist at all
# This prevents resetting them on every rerun unless explicitly done by mode change
if "agraph_nodes" not in st.session_state:
    st.session_state.agraph_nodes = [Node(id=var, label=var, shape="box", style="rounded") for var in VARIABLES]
if "agraph_edges" not in st.session_state:
    st.session_state.agraph_edges = []

# --- Network Management --- 
col1, col2, col3 =  st.columns([1, 0.1, 1])

with col1:
    st.subheader("Manage Expert Networks")
    st.caption("*Ensure that the uploaded graphs are DAGs, i.e., they have no directed cycles.*")

    # Check existing files for upload constraints
    existing_networks = get_expert_network_files()
    high_exists = "High Confidence" in existing_networks
    moderate_exists = "Moderate Confidence" in existing_networks

    # 1. Upload Graphs
    st.markdown("**Upload Specific Confidence Graphs:**")
    rerun_needed_after_upload = False

    # High Confidence Upload
    uploaded_high = st.file_uploader("Upload High Confidence Graph (.csv)", type=["csv"], key="upload_high")
    if uploaded_high is not None:
        try:
            if uploaded_high.name.endswith('.csv'):
                # Handle CSV upload
                csv_data = uploaded_high.getvalue().decode('utf-8')
                parsed_nodes, parsed_edges = parse_graph_data(csv_data)
                
            if save_graph_to_file("High Confidence", parsed_nodes, parsed_edges):
                st.success("High Confidence graph uploaded and saved.")
                st.session_state.last_selected_action = "Select Action..."
                rerun_needed_after_upload = True
            else:
                st.error("Failed to save uploaded High Confidence graph.")
        except Exception as e:
            st.error(f"Error processing High Confidence upload: {e}")
        if rerun_needed_after_upload: st.rerun()

    # Moderate Confidence Upload
    uploaded_moderate = st.file_uploader("Upload Moderate Confidence Graph (.csv)", type=["csv"], key="upload_moderate", disabled=not high_exists)
    if uploaded_moderate is not None and high_exists:
        try:
            if uploaded_moderate.name.endswith('.csv'):
                # Handle CSV upload
                csv_data = uploaded_moderate.getvalue().decode('utf-8')
                parsed_nodes, parsed_edges = parse_graph_data(csv_data)
                
            if save_graph_to_file("Moderate Confidence", parsed_nodes, parsed_edges):
                st.success("Moderate Confidence graph uploaded and saved.")
                st.session_state.last_selected_action = "Select Action..."
                rerun_needed_after_upload = True
            else:
                st.error("Failed to save uploaded Moderate Confidence graph.")
        except Exception as e:
            st.error(f"Error processing Moderate Confidence upload: {e}")
        if rerun_needed_after_upload: st.rerun()
    elif not high_exists:
        st.caption("Upload High Confidence graph first to enable Moderate upload.")

    # Low Confidence Upload
    uploaded_low = st.file_uploader("Upload Low Confidence Graph (.csv)", type=["csv"], key="upload_low", disabled=not moderate_exists)
    if uploaded_low is not None and moderate_exists:
        try:
            if uploaded_low.name.endswith('.csv'):
                # Handle CSV upload
                csv_data = uploaded_low.getvalue().decode('utf-8')
                parsed_nodes, parsed_edges = parse_graph_data(csv_data)
                
            if save_graph_to_file("Low Confidence", parsed_nodes, parsed_edges):
                st.success("Low Confidence graph uploaded and saved.")
                st.session_state.last_selected_action = "Select Action..."
                rerun_needed_after_upload = True
            else:
                st.error("Failed to save uploaded Low Confidence graph.")
        except Exception as e:
            st.error(f"Error processing Low Confidence upload: {e}")
        if rerun_needed_after_upload: st.rerun()
    elif not moderate_exists:
        st.caption("Upload Moderate Confidence graph first to enable Low upload.")

with col2:
    st.markdown(
        """
        <style>
        .vertical-line {
            border-left: 1px solid #d3d3d3;  # light grey;
            height: 545px;
            margin-left: 20px;
            margin-right: 2px;
            margin-top: auto;
            margin-bottom: auto;
        }
        </style>

        <div class="vertical-line"></div>
        """,
        unsafe_allow_html=True
    )
with col3:
    # 2. Select Network to Load/Define
    st.markdown("**Load Existing or Define New Network:**")
    available_networks = get_expert_network_files() # Gets standard names

    options = ["Select Action..."]
    if available_networks:
        options.extend(available_networks)
    options.append("Define New Network")

    # Determine the index for the selectbox based on the last action
    current_action = st.session_state.last_selected_action
    try:
        select_index = options.index(current_action)
    except ValueError:
        select_index = 0 # Default to "Select Action..."

    selected_action = st.selectbox(
        "Action:",
        options=options,
        key="network_action_selector",
        index=1 if len(options)>2 else select_index # Set index to maintain selection across reruns unless changed
    )

    # --- Mode Switching Logic --- 
    # Only change mode and graph state IF the selection has actually changed
    if selected_action != st.session_state.last_selected_action:
        st.session_state.last_selected_action = selected_action # Update the tracker
        needs_rerun = False
        if selected_action == "Define New Network":
            if st.session_state.graph_mode != "define":
                st.session_state.graph_mode = "define"
                st.session_state.loaded_network_name = None
                # Explicitly reset graph state for Define New mode
                st.session_state.agraph_nodes = [Node(id=var, label=var, shape="box", style="rounded") for var in VARIABLES]
                st.session_state.agraph_edges = []
                st.info("Switched to Define Mode. Add edges using the editor.")
                needs_rerun = True
        elif selected_action != "Select Action...":
            # Load the selected network for viewing
            if st.session_state.graph_mode != "view" or st.session_state.loaded_network_name != selected_action:
                st.session_state.graph_mode = "view"
                st.session_state.loaded_network_name = selected_action
                # Load graph state from file
                loaded_nodes, loaded_edges = load_graph_from_file(selected_action)
                st.session_state.agraph_nodes = loaded_nodes
                st.session_state.agraph_edges = loaded_edges
                st.success(f"Loaded \'{selected_action}\' for viewing.")
                needs_rerun = True
        else: # Selected "Select Action..."
            if st.session_state.graph_mode != "none":
                st.session_state.graph_mode = "none"
                st.session_state.loaded_network_name = None
                # Reset to default nodes, empty edges
                st.session_state.agraph_nodes = [Node(id=var, label=var, shape="box", style="rounded") for var in VARIABLES]
                st.session_state.agraph_edges = []
                needs_rerun = True
        
        # Rerun only if the mode or loaded graph actually changed
        if needs_rerun:
            st.rerun()
    st.markdown("--- ")
    st.subheader("Graph Information")
    # Show node/edge counts (using the potentially updated state)
    node_count = len(st.session_state.get("agraph_nodes", []))
    edge_count = len(st.session_state.get("agraph_edges", []))
    
    with st.container(border=True):
        info_cols = st.columns(2)
        info_cols[0].metric("Nodes", node_count)
        info_cols[1].metric("Edges", edge_count)

st.markdown("--- ")
st.subheader("Graph Editor")

# Determine editability based on the current mode
is_editable = (st.session_state.graph_mode == "define")

# Display current mode status
if st.session_state.graph_mode == "view":
    st.info(f"Viewing: **{st.session_state.loaded_network_name}** (View-Only)")
elif st.session_state.graph_mode == "define":
    st.info("**Define New Network Mode** (Editable)")
    st.caption("*Ensure that no cycles are created. Currently, it only prevents reverse- and self-edges.*")
else:
    st.warning("Select an action: Load a network to view or choose 'Define New'.")
# --- Interactive Graph Editor --- 
if is_editable:
    st.markdown("**Edge Confidence for New Edges:**")
    edge_confidence = st.selectbox(
        "Select Confidence:",
        options=["High", "Moderate", "Low"],
        key="edge_type_selector" # Key used by graph_editor.py
    )
    st.caption("Use editor controls (buttons on graph) to add/delete edges.")
else:
    st.caption("Graph is in view-only mode. Cannot save changes.")

# Rerun if the editor signaled that an edge was added or deleted
state_changed = display_graph_with_manual_controls()
if state_changed:
    # Force a rerun to update the UI
    st.rerun()

# --- Save Network Button (only for Define mode) --- 
if is_editable:
    st.markdown("--- ")
    if st.button("Save Defined Network(s)", type="primary", key="save_defined_button"):
        # Read the latest state directly from session_state when button is clicked
        nodes_to_save = st.session_state.get("agraph_nodes", [])
        edges_to_save = st.session_state.get("agraph_edges", [])
        
        # Use the dedicated function from graph_editor.py for Rule 6
        # This function now handles the empty edge check internally
        if save_defined_network(nodes_to_save, edges_to_save):
            st.success("Defined network(s) saved based on confidence rules.")
            # Reset mode after saving to force user to select action again
            st.session_state.graph_mode = "none" 
            st.session_state.last_selected_action = "Select Action..." 
            st.rerun()
        else:
            # Error messages (like empty graph) are handled within save_defined_network
            pass
st.markdown("---")