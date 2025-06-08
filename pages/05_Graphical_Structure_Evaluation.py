import streamlit as st
import torch
torch.classes.__path__ = [] # adding this line to manually avoid "examining the path of torch.classes raised"; some streamlit internal error, doesn't affect app.
from modules.graph_editor import display_graph_editor
from modules.evaluation_metrics import evaluate_dag_metrics
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.bif_loader import load_bif_network, load_all_models
from modules.model_options import model_options
import logging

# Configure logging to suppress pgmpy INFO messages
logging.getLogger('pgmpy').setLevel(logging.WARNING)

st.set_page_config(page_title="Graphical Structure Evaluation", page_icon="↔️", layout="wide")


st.title("↔️ Graphical Structure Evaluation")
st.markdown("Visually compare different causal models and evaluate the structural similarity across varying expert knowledge graphs.")
st.caption("Note: The following results are based on the preloaded learned structures and expert knowledge graphs. These can be overwritten.")
st.markdown("---")

st.subheader("Visually Inspect the Structure of Two Selected Models")
col1, col2 = st.columns(2)
with col1:
    selected_model1 = st.selectbox("Select Model 1:", model_options, key="eval_model1",index=7)
    # Load and display the selected model from BIF
    nodes1, edges1, model1 = load_bif_network(selected_model1)
    if nodes1 and edges1:
        st.session_state.agraph_nodes = nodes1
        st.session_state.agraph_edges = edges1
        display_graph_editor(editable=False)
        
with col2:
    selected_model2 = st.selectbox("Select Model 2:", model_options, key="eval_model2",index=8)
    if selected_model2 != selected_model1:
        # Load and display the selected model from BIF
        nodes2, edges2, model2 = load_bif_network(selected_model2)
        if nodes2 and edges2:
            st.session_state.agraph_nodes = nodes2
            st.session_state.agraph_edges = edges2
            display_graph_editor(editable=False)
    else:
        st.warning("Select two different models (Model 1 and Model 2) above to compare their structures.")

if selected_model1 != selected_model2:
    st.subheader(f"Metric Based Comparison: {selected_model1} vs {selected_model2}")

    if model1 and model2:
        # Calculate structural metrics
        metrics = evaluate_dag_metrics(model1, model2, metrics=['SHD', 'F1', 'BSF'])
            
        with st.container(border=False):
            scol1, scol2, scol3 = st.columns(3)
            scol1.metric("SHD", f"{metrics['SHD']}", delta_color="inverse", help="Structural Hamming Distance (Lower is better)")
            scol2.metric("F1 Score", f"{metrics['F1']:.3f}", help="Balance of Precision/Recall (Higher is better)")
            scol3.metric("BSF Score", f"{metrics['BSF']:.3f}", help="Balanced Scoring Function (Higher is better)")
    else:
        st.error("Could not load one or both models for comparison.")
else:
    st.warning("Select two different models (Model 1 and Model 2) above to compare their structures.")
    
st.markdown("---")
st.subheader("Comparing Learned Structures Across Expert Knowledge Graphs")
    
# Load models
learned_models, knowledge_graphs, loaded_models = load_all_models()

# Color map
colors = {
    "Peter-Clark Algorithm": "blue",
    "Greedy Equivalence Search": "green",
    "Hill Climbing Search": "red",
    "Simulated Annealing": "purple",
    "Evolutionary Algorithm": "orange",
    "Multi-Agent Genetic Algorithm": "pink",
    "Particle Swarm Optimization": "brown",
    "Averaged Structure": "gray"
    }


fig = make_subplots(rows=1, cols=3, subplot_titles=list(knowledge_graphs.keys()))
col_index = 1

for kg_name in knowledge_graphs.keys():
    kg_model = loaded_models[kg_name]

    for model_name in learned_models.keys():
        model = loaded_models[model_name]
        metrics = evaluate_dag_metrics(kg_model, model, metrics=['SHD', 'F1', 'BSF'])
        avg_score = (metrics['BSF'] + metrics['F1']) / 2

        fig.add_trace(
            go.Scatter(
                x=[avg_score],
                y=[metrics['SHD']],
                mode='markers',
                name=model_name if col_index == 1 else None,  # avoid repeat legend
                marker=dict(color=colors[model_name], size=10),
                showlegend=(col_index == 1)
                ),row=1,col=col_index
                )

    # Add gridlines (manual, since Plotly doesn't do matplotlib-style grids)
    fig.update_xaxes(title_text="Avg(BSF, F1)", zeroline=True, zerolinecolor='gray', row=1, col=col_index)
    fig.update_yaxes(title_text="SHD", zeroline=True, zerolinecolor='gray', row=1, col=col_index)

    col_index += 1

fig.update_layout(
    title="Comparison of SHD and Avg (BSF, F1) of Learned Structures Against Knowledge Graphs",
    height=500,
    width=1000,
    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
        

st.plotly_chart(fig, use_container_width=True)
st.caption("The SHD and the average of BSF+F1 scores produced by the structure learning algorithms and the model-averaging graph, with reference to the three knowledge graphs reflecting the three different confidence levels.")

# Compute comparison metrics for bar charts
confidence_levels = list(knowledge_graphs.keys())  # e.g., ["High Confidence", "Moderate Confidence", "Low Confidence"]
# We will use simplified labels for the x-axis:
x_labels = ["High", "Moderate", "Low"]

# Prepare lists to store values for each confidence level
learned_shd_list = []
averaged_shd_list = []
learned_avg_list = []
averaged_avg_list = []

for conf in confidence_levels:
    # Get the knowledge graph model for this confidence level
    kg_model = loaded_models[conf]

    # Compute metrics for learned algorithms (excluding "Averaged Structure")
    shd_values = []
    avg_score_values = []
    for model_name in learned_models.keys():
        if model_name != "Averaged Structure":
            metrics = evaluate_dag_metrics(kg_model, loaded_models[model_name], metrics=['SHD', 'F1', 'BSF'])
            shd_values.append(metrics['SHD'])
            avg_score_values.append((metrics['F1'] + metrics['BSF'])/2)
    # Compute average over these learned models
    avg_learned_shd = sum(shd_values) / len(shd_values)
    avg_learned_avg = sum(avg_score_values) / len(avg_score_values)
    
    # Compute metrics for the "Averaged Structure"
    metrics_avg_struct = evaluate_dag_metrics(kg_model, loaded_models["Averaged Structure"], metrics=['SHD', 'F1', 'BSF'])
    averaged_shd = metrics_avg_struct['SHD']
    averaged_avg = (metrics_avg_struct['F1'] + metrics_avg_struct['BSF']) / 2

    # Append computed values to lists
    learned_shd_list.append(avg_learned_shd)
    averaged_shd_list.append(averaged_shd)
    learned_avg_list.append(avg_learned_avg)
    averaged_avg_list.append(averaged_avg)


# Create a second Plotly figure with two bar chart subplots
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "Comparison of Average SHD of Learned Models and Averaged Structure SHD",
        "Comparison of Avg(BSF, F1) of Learned Models and Averaged Structure Avg(BSF, F1)"
        ]
        )

# First subplot: SHD comparison
fig2.add_trace(
    go.Bar(
        x=x_labels,
        y=learned_shd_list,
        name="Average SHD for Learned Algos",
        marker_color="skyblue"
        ),
        row=1,
        col=1
        )

fig2.add_trace(
    go.Bar(
        x=x_labels,
        y=averaged_shd_list,
        name="SHD for Averaged Structure",
        marker_color="orange"
        ),
        row=1,
        col=1
        )

# Second subplot: Avg(BSF, F1) comparison
fig2.add_trace(
    go.Bar(
        x=x_labels,
        y=learned_avg_list,
        name="Average Avg(BSF, F1) for Learned Algos",
        marker_color="cornflowerblue"
        ),
        row=1,
        col=2
        )

fig2.add_trace(
    go.Bar(
        x=x_labels,
        y=averaged_avg_list,
        name="Avg(BSF, F1) for Averaged Structure",
        marker_color="tomato"
        ),
        row=1,
        col=2
        )

# Update layout: setting grouped bars, titles, etc.
fig2.update_layout(
    barmode='group',
    title_text="Comparison of SHD and Avg(BSF, F1) Between Learned Models and Averaged Structure Across Expert Knowledge Graphs",
    width=1000,
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )

# Display the new chart
st.plotly_chart(fig2, use_container_width=True)
st.caption("The SHD and the average of BSF+F1 scores produced by the structure learning algorithms and the model-averaging graph, with reference to the three knowledge graphs reflecting the three different confidence levels.")
st.markdown("---")