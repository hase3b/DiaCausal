import streamlit as st
import torch
torch.classes.__path__ = [] # adding this line to manually avoid "examining the path of torch.classes raised"; some streamlit internal error, doesn't affect app.
import os
import time

st.set_page_config(page_title="Evidence Based Learning", page_icon="💡", layout="wide")

st.title("💡 Evidence Based Learning")
st.markdown("""
            Learn causal graph structures directly from the data using various algorithms.
            """)
st.caption("Note: Currently, the application is preloaded with (parameterized) learned structures given default hyperparameters. These can be overwritten.")
st.markdown("---")
st.subheader("Structure Learning Algorithms")
with st.container(border=True):
    # First row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Peter-Clark Algorithm**")
    with col2:
        st.markdown("**Greedy Equivalence Search**")
    with col3:
        st.markdown("**Hill Climbing Search**")
    with col4:
        st.markdown("**Simulated Annealing**")
    
    # Second row
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown("**Evolutionary Algorithm**")
    with col6:
        st.markdown("**Multi-Agent Genetic Algorithm**")
    with col7:
        st.markdown("**Particle Swarm Optimization**")
    with col8:
        st.markdown("**Averaged Structure**")

# --- Algorithm Configuration ---
st.subheader("Configure Learning Parameters")

# Initial DAG Selection
initial_dag_options = [
    "Empty",
    "Random",
    "Sparse Random",
    "Tree (TAN)",
    "Tree (Chow-Liu)"
]
selected_initial_dag = st.selectbox(
    "Initial DAG (Where Applicable):",
    initial_dag_options,
    index=1  # Default to Random
)

# Scoring Method Selection
scoring_method_options = [
    "BIC",
    "AIC",
    "K2",
    "BDeu",
    "BDs"
]
selected_scoring = st.selectbox(
    "Scoring Method (Where Applicable):",
    scoring_method_options,
    index=0  # Default to BIC
)

# Minimum Edge Frequency Input
min_edge_freq = st.number_input(
    "Averaged Structure - Minimum Edge Frequency:",
    min_value=1,
    max_value=7,
    value=4,
    step=1
)

# --- Run Learning ---
st.subheader("Structure Learning")
st.caption("*This may take a while to run. Estimated time: 15 minutes.*")
if st.button("Learn Structures"):
    st.info("Running structure learning algorithms...")
    
    # Read the data
    try:
        start_time = time.time()
        
        # Lazy load dependencies only when needed
        from modules.learn_structure import learn_network, generate_initial_dag, read_data
        
        data = read_data("data/diabetescategorised.csv")
        variables = list(data.columns)
        
        # Generate initial DAG based on user selection
        initial_dag = generate_initial_dag(
            data=data,
            method=selected_initial_dag.lower().replace(" ", "_"),
            sparsity=0.2
        )
        
        if selected_scoring == "BIC":
            score_type = "bic-d"
        elif selected_scoring == "AIC":
            score_type = "aic-d"
        elif selected_scoring == "K2":
            score_type = "k2"
        elif selected_scoring == "BDeu":
            score_type = "bdeu"
        elif selected_scoring == "BDs":
            score_type = "bds"

        # Run structure learning
        learn_network(
            initial_dag=initial_dag,
            data=data,
            variables=variables,
            score_type=score_type,
            min_edge_frequency=min_edge_freq
            )
        
        end_time = time.time()
        duration = end_time - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        st.success(f"Structure learning complete and results saved! (Time taken: {minutes}m {seconds}s)")
        st.success("Ensure to run parameter learning again if relearned structures otherwise further parts won't work as expected!")
    except Exception as e:
        st.error(f"An error occurred during structure learning: {str(e)}")

# --- Parameter Learning ---
st.subheader("Parameter Learning (Incld. Expert Graphs)")
# Check if evidence-based networks exist
evidence_based_dir = "networks/evidence_based"
has_evidence_networks = os.path.exists(evidence_based_dir) and len(os.listdir(evidence_based_dir)) > 0

if has_evidence_networks:
    # Initial DAG Selection
    estimator_options = [
        "MLE",
        "Bayes",
        ]
    selected_estimator = st.selectbox(
        "Estimator:",
        estimator_options,
        index=0  # Default to MLE
        )
    if selected_estimator == "Bayes":
        equivalent_sample_size=st.number_input(
            "Equivalent Sample Size (Uniform Prior):",
            min_value=1,
            max_value=100,
            value=10,
            step=1
            )

    if st.button("Learn Parameters"):
        from modules.learn_parameters import param_learning
        import time
        
        start_time = time.time()
        param_learning(data_path="data/diabetescategorised.csv", input_dir="networks/evidence_based", output_dir="networks/cbns", estimator_type=selected_estimator, equivalent_sample_size=equivalent_sample_size if selected_estimator == "Bayes" else None)
        param_learning(data_path="data/diabetescategorised.csv", input_dir="networks/expert_knowledge", output_dir="networks/cbns", estimator_type=selected_estimator, equivalent_sample_size=equivalent_sample_size if selected_estimator == "Bayes" else None)
        end_time = time.time()
        
        time_taken = end_time - start_time
        st.success(f"Parameter learning complete and results saved! (Time taken: {time_taken:.2f}s)")
        st.cache_data.clear()
else:
    st.warning("No evidence-based networks found. Please run structure learning first.")

st.markdown("--- ")
st.caption("The networks learned here can be compared with expert knowledge and be used for causal reasoning on the pages that follow.")