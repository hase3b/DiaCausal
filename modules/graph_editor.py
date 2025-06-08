import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import os
import json
import csv
from io import StringIO

# Define variables based on the paper
VARIABLES = [
    'Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
    'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
    'Sex', 'Age', 'Education', 'Income'
]

EXPERT_NETWORK_DIR = os.path.join(os.path.dirname(__file__), "..", "networks", "expert_knowledge")
NETWORK_FILENAMES = {
    "High Confidence": "high_confidence_knowledge_graph.json",
    "Moderate Confidence": "moderate_confidence_knowledge_graph.json",
    "Low Confidence": "low_confidence_knowledge_graph.json"
}

def initialize_graph_state(nodes=None, edges=None):
    """Initializes or updates the graph structure in session state."""
    
    # Initialize nodes if not in session state
    if "agraph_nodes" not in st.session_state:
        st.session_state.agraph_nodes = [Node(id=var, label=var, shape="box", style="rounded") for var in VARIABLES]
    
    # Initialize edges if not in session state
    if "agraph_edges" not in st.session_state:
        st.session_state.agraph_edges = []
    
    # Update nodes if provided
    if nodes is not None:
        st.session_state.agraph_nodes = nodes
    
    # Update edges if provided
    if edges is not None:
        st.session_state.agraph_edges = edges
    
    if "selected_edge" not in st.session_state:
        st.session_state.selected_edge = None

def get_expert_network_files():
    """Checks for existing expert network files and returns their standard names."""
    os.makedirs(EXPERT_NETWORK_DIR, exist_ok=True)
    available_networks = []
    for display_name, filename in NETWORK_FILENAMES.items():
        if os.path.exists(os.path.join(EXPERT_NETWORK_DIR, filename)):
            available_networks.append(display_name)
    # Ensure order High -> Moderate -> Low if they exist
    ordered_available = []
    if "High Confidence" in available_networks: ordered_available.append("High Confidence")
    if "Moderate Confidence" in available_networks: ordered_available.append("Moderate Confidence")
    if "Low Confidence" in available_networks: ordered_available.append("Low Confidence")
    return ordered_available

def get_filename_from_display_name(display_name):
    """Converts standard display name back to specific filename."""
    return NETWORK_FILENAMES.get(display_name)

def save_graph_to_file(graph_display_name, nodes, edges):
    """Saves the graph structure (nodes and edges) to the specific JSON file."""
    filename = get_filename_from_display_name(graph_display_name)
    if not filename:
        st.error(f"Invalid graph name for saving: {graph_display_name}")
        return False

    filepath = os.path.join(EXPERT_NETWORK_DIR, filename)
    os.makedirs(EXPERT_NETWORK_DIR, exist_ok=True)

    # Ensure nodes include all VARIABLES
    node_ids = {node.id for node in nodes}
    all_nodes_data = [node.to_dict() for node in nodes]
    for var in VARIABLES:
        if var not in node_ids:
            all_nodes_data.append(Node(id=var, label=var, shape="box", style="rounded").to_dict())

    graph_data = {
        "nodes": all_nodes_data,
        "edges": [edge.to_dict() for edge in edges]
    }

    try:
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving graph to {filepath}: {e}")
        return False

def parse_csv_graph_data(csv_data):
    """Parses CSV graph data into Node and Edge objects."""
    # Create default nodes for all variables
    nodes = [Node(id=var, label=var, shape="box", style="rounded", size=35) for var in VARIABLES]
    edges = []
    
    # Parse CSV data
    csv_file = StringIO(csv_data)
    reader = csv.DictReader(csv_file)
    
    for row in reader:
        source = row['Variable 1'].strip()  # Remove any whitespace
        target = row['Variable 2'].strip()  # Remove any whitespace
        
        # Skip if source or target is not in VARIABLES
        if source not in VARIABLES or target not in VARIABLES:
            continue
            
        # Create edge with default confidence (Low)
        edge = Edge(
            source=source,
            target=target,
            style="solid",
            width=2  # Increased width for better visibility
        )
        edges.append(edge)
    
    return nodes, edges

def parse_graph_data(graph_data):
    """Parses graph data (CSV string) into Node and Edge objects."""
    # Check if input is CSV data
    if isinstance(graph_data, str) and ',' in graph_data and '\n' in graph_data:
        return parse_csv_graph_data(graph_data)
        
    # Handle JSON format
    loaded_nodes_data = graph_data.get('nodes', [])
    loaded_edges_data = graph_data.get('edges', [])

    # Create Node objects
    loaded_nodes = [Node(id=n['id'],
                       label=n.get('label', n['id']),
                       shape=n.get('shape', 'box'),
                       style=n.get('style', 'rounded'))
                    for n in loaded_nodes_data]

    # Ensure all VARIABLES are present as nodes
    existing_node_ids = {n.id for n in loaded_nodes}
    for var in VARIABLES:
        if var not in existing_node_ids:
            loaded_nodes.append(Node(id=var, label=var, shape="box", style="rounded", size=35))

    # Create Edge objects
    loaded_edges = []
    for e in loaded_edges_data:
        # Handle both source/from and target/to variations
        source = e.get('source') or e.get('from')
        target = e.get('target') or e.get('to')
        
        if not source or not target:
            continue  # Skip edges with missing source or target

        loaded_edges.append(Edge(
            source=source,
            target=target,
            style=e.get('style', 'solid'),
            width=e.get('width', 2)
        ))

    return loaded_nodes, loaded_edges

def load_graph_from_file(graph_display_name):
    """Loads a graph structure from the specific JSON file."""
    filename = get_filename_from_display_name(graph_display_name)
    if not filename:
        st.error(f"Invalid graph name for loading: {graph_display_name}")
        # Return default empty graph if name is invalid
        return [Node(id=var, label=var, shape="box", style="rounded") for var in VARIABLES], []

    filepath = os.path.join(EXPERT_NETWORK_DIR, filename)

    if not os.path.exists(filepath):
        st.warning(f"Graph file not found: {filepath}. Cannot load.")
        # Return default empty graph if file not found
        return [Node(id=var, label=var, shape="box", style="rounded") for var in VARIABLES], []

    try:
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        return parse_graph_data(graph_data)

    except Exception as e:
        st.error(f"Error loading graph from {filepath}: {e}")
        # Return default empty graph on error
        return [Node(id=var, label=var, shape="box", style="rounded") for var in VARIABLES], []

def delete_edge(edge_id):
    """Deletes a specific edge from session state."""
    edges = st.session_state.agraph_edges
    # Handle both formats: "source_target" and {"source": source, "target": target}
    if isinstance(edge_id, dict):
        source = edge_id.get('source')
        target = edge_id.get('target')
        if source and target:
            new_edges = [edge for edge in edges if not (edge.source == source and edge.target == target)]
        else:
            return False
    else:
        # Handle string format "source_target"
        try:
            source, target = edge_id.split('_')
            new_edges = [edge for edge in edges if not (edge.source == source and edge.target == target)]
        except:
            return False

    deleted = len(new_edges) < len(edges)
    if deleted:
        st.session_state.agraph_edges = new_edges
        st.session_state.selected_edge = None
    return deleted

def add_edge_with_properties(source_node, target_node, confidence="Low"):
    """Adds a new edge with specified confidence."""
    
    # Check if edge already exists in either direction using getattr to handle both attribute styles
    for edge in st.session_state.agraph_edges:
        edge_source = getattr(edge, 'source', getattr(edge, 'from', None))
        edge_target = getattr(edge, 'target', getattr(edge, 'to', None))
        
        # Check both directions
        if ((edge_source == source_node and edge_target == target_node) or
            (edge_source == target_node and edge_target == source_node)):
            st.warning(f"Edge between {source_node} and {target_node} already exists in one direction.")
            return False
    
    # Create new edge with consistent attribute names
    new_edge = Edge(
        source=source_node,
        target=target_node,
        style="solid",
        width=3,
        data={"confidence": confidence}
    )
    
    # Create a new list with the new edge
    st.session_state.agraph_edges = st.session_state.agraph_edges + [new_edge]
    
    return True

def get_agraph_config(editable=True):
    """Returns the configuration for the agraph component, allowing edit mode control."""
    config = Config(
        width='100%',
        height=600,
        directed=True,
        hierarchical=False,
        # Removed invalid options that were causing vis.js errors
        interaction={
            'navigationButtons': True,
            'tooltipDelay': 300,
            'selectConnectedEdges': True,
            'hover': True,
            'dragNodes': True,
            'dragView': True,
            'zoomView': True
        },
        manipulation={
            'enabled': editable,
            'initiallyActive': editable,
            'addNode': False,
            'addEdge': editable,
            'editEdge': False,
            'deleteNode': False,
            'deleteEdge': editable
        },
        physics={
            'enabled': True,
            'solver': 'forceAtlas2Based',
            'forceAtlas2Based': {
                'gravitationalConstant': -50,
                'centralGravity': 0.01,
                'springLength': 200,
                'springConstant': 0.08,
                'damping': 0.4,
                'avoidOverlap': 1
            },
            'stabilization': {
                'enabled': True,
                'iterations': 1000,
                'updateInterval': 25,
                'onlyDynamicEdges': False,
                'fit': True
            }
        },
        # Fixed: Corrected node and edge configuration
        nodes={
            'shape': 'box',
            'font': {'color': '#000000'},
            'borderWidth': 2,
            'shadow': True
        },
        edges={
            'arrows': 'to',
            'smooth': False,
            'font': {'color': '#000000', 'strokeWidth': 2, 'strokeColor': '#ffffff'}
        }
    )
    return config

def display_graph_editor(editable=True):
    """Displays the interactive graph editor and handles interactions."""
    
    # Initialize state if needed
    if "agraph_edges" not in st.session_state:
        initialize_graph_state()
    
    config = get_agraph_config(editable=editable)

    # Get the selected edge type for adding new edges (only relevant if editable)
    selected_edge_type = st.session_state.get("edge_type_selector", "High")

    # Display the graph
    try:
        graph_state_return = agraph(
            nodes=st.session_state.agraph_nodes,
            edges=st.session_state.agraph_edges,
            config=config
        )
        
        # Handle interactions ONLY if editable
        if editable and graph_state_return:
            
            # Check for different types of interactions
            if isinstance(graph_state_return, dict):
                # Handle edge addition
                if 'edges' in graph_state_return and graph_state_return['edges']:
                    # New edge was created
                    new_edges = graph_state_return['edges']
                    
                    # Process new edges
                    for edge_data in new_edges:
                        if isinstance(edge_data, dict) and 'from' in edge_data and 'to' in edge_data:
                            source = edge_data['from']
                            target = edge_data['to']
                            if add_edge_with_properties(source, target, selected_edge_type):
                                st.success(f"Added edge: {source} -> {target}")
                                st.rerun()
                
                # Handle edge deletion
                if 'deleted' in graph_state_return and graph_state_return['deleted']:
                    deleted_items = graph_state_return['deleted']
                    
                    if 'edges' in deleted_items:
                        for edge_id in deleted_items['edges']:
                            if delete_edge(edge_id):
                                st.success(f"Deleted edge: {edge_id}")
                                st.rerun()
            
            # Handle direct edge creation callback
            elif hasattr(graph_state_return, 'edges'):
                current_edges = [f"{e.source}_{e.target}" for e in st.session_state.agraph_edges]
                returned_edges = graph_state_return.edges if graph_state_return.edges else []
                
                # Check for new edges
                for edge_id in returned_edges:
                    if edge_id not in current_edges:
                        try:
                            source, target = edge_id.split('_')
                            if add_edge_with_properties(source, target, selected_edge_type):
                                st.success(f"Added edge: {source} -> {target}")
                                st.rerun()
                        except ValueError:
                            st.warning(f"Could not parse edge ID: {edge_id}")
        
    except Exception as e:
        st.error(f"Error displaying graph: {str(e)}")
        return False

    return True

# Alternative approach using manual edge addition controls
def display_graph_with_manual_controls():
    """Display graph with manual controls for edge addition/removal."""
    
    # Initialize state if needed
    if "agraph_edges" not in st.session_state:
        initialize_graph_state()
    
    # Manual edge addition controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_node = st.selectbox("Source Node", VARIABLES, key="manual_source")
    
    with col2:
        target_node = st.selectbox("Target Node", VARIABLES, key="manual_target")
    
    with col3:
        confidence = st.selectbox("Confidence", ["High", "Moderate", "Low"], key="manual_confidence")
    
    # Add and Delete edge buttons in a row
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Add Edge", use_container_width=True):
            if source_node != target_node:
                if add_edge_with_properties(source_node, target_node, confidence):
                    st.success(f"Added edge: {source_node} -> {target_node} ({confidence})")
                    st.rerun()
                else:
                    st.warning("Edge already exists or couldn't be added")
            else:
                st.warning("Source and target nodes must be different")
    
    with col2:
        if st.button("Delete Edge", use_container_width=True):
            if source_node != target_node:
                # Check if edge exists in either direction
                edge_exists = False
                edge_to_delete = None
                for edge in st.session_state.agraph_edges:
                    edge_source = getattr(edge, 'source', getattr(edge, 'from', None))
                    edge_target = getattr(edge, 'target', getattr(edge, 'to', None))
                    if ((edge_source == source_node and edge_target == target_node) or
                        (edge_source == target_node and edge_target == source_node)):
                        edge_exists = True
                        edge_to_delete = edge
                        break
                
                if edge_exists and edge_to_delete:
                    # Remove the edge from the list
                    st.session_state.agraph_edges = [e for e in st.session_state.agraph_edges if e != edge_to_delete]
                    st.success(f"Deleted edge between {source_node} and {target_node}")
                    st.rerun()
                else:
                    st.warning(f"No edge exists between {source_node} and {target_node}")
            else:
                st.warning("Source and target nodes must be different")
    
    # Display the graph (read-only)
    config = get_agraph_config(editable=False)
    try:
        agraph(
            nodes=st.session_state.agraph_nodes,
            edges=st.session_state.agraph_edges,
            config=config
        )
    except Exception as e:
        st.error(f"Error displaying graph: {str(e)}")

# --- Rule 6 Implementation --- 
def filter_edges(edges, allowed_confidences):
    """Filters edges based on a list of allowed confidence levels."""
    return [edge for edge in edges if edge.data.get("confidence") in allowed_confidences]

def save_defined_network(nodes, edges):
    """Saves the defined network based on Rule 6."""
    if not edges:
        st.warning("Cannot save an empty graph. Please add edges.")
        return False # Indicate failure

    confidences_present = {edge.data.get("confidence") for edge in edges}
    saved_something = False
    all_saves_successful = True

    # Determine which networks to save based on present confidences
    save_high = False
    save_moderate = False
    save_low = False

    if "High" in confidences_present:
        save_high = True
        save_moderate = True
        save_low = True
    elif "Moderate" in confidences_present:
        save_moderate = True
        save_low = True
    elif "Low" in confidences_present:
        save_low = True

    # Save High Confidence Network
    if save_high:
        high_only_edges = filter_edges(edges, ["High"])
        if save_graph_to_file("High Confidence", nodes, high_only_edges):
            st.success("High Confidence network saved (High edges only).")
            saved_something = True
        else:
            st.error("Failed to save High Confidence network.")
            all_saves_successful = False

    # Save Moderate Confidence Network
    if save_moderate:
        high_moderate_edges = filter_edges(edges, ["High", "Moderate"])
        if save_graph_to_file("Moderate Confidence", nodes, high_moderate_edges):
            st.success("Moderate Confidence network saved (High & Moderate edges).")
            saved_something = True
        else:
            st.error("Failed to save Moderate Confidence network.")
            all_saves_successful = False

    # Save Low Confidence Network
    if save_low:
        all_edges = filter_edges(edges, ["High", "Moderate", "Low"]) # Essentially all edges
        if save_graph_to_file("Low Confidence", nodes, all_edges):
            st.success("Low Confidence network saved (All edges).")
            saved_something = True
        else:
            st.error("Failed to save Low Confidence network.")
            all_saves_successful = False

    # Return True only if at least one save happened AND all attempted saves were successful
    return saved_something and all_saves_successful