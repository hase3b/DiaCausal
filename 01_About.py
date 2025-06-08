import streamlit as st
import torch
torch.classes.__path__ = [] # adding this line to manually avoid "examining the path of torch.classes raised"; some streamlit internal error, doesn't affect app.
from modules.header import show_header

st.set_page_config(page_title="About", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

show_header()
st.markdown("---")
st.title("🩺 About This Application")

st.markdown("""
## Project Description
            
Non-communicable diseases (NCDs) such as cardiovascular conditions, cancer, and diabetes are the leading causes of global mortality, 
highlighting the need for effective prevention and management strategies. While Artificial Intelligence (AI) holds great promise in 
healthcare for tasks like diagnosis and prediction, its lack of transparency and causal understanding limits its utility in clinical 
decision-making. Integrating causal inference into AI models addresses this gap by enabling interpretable, intervention-driven insights 
that support more informed and effective healthcare outcomes.

This Streamlit application serves as an interactive decision support system, developed based on the research presented in:
            
    Zahoor, S., Constantinou, A. C., Curtis, T. M., & Hasanuzzaman, M. (2024b). Investigating the validity of structure
            learning algorithms in identifying risk factors for intervention in patients with diabetes.
            arXiv (Cornell University). https://doi.org/10.48550/arxiv.2403.14327

The goal is to provide a platform for doctors, researchers, and medical students to:
*   Explore the relationships between various risk factors and diabetes using the BRFSS 2015 dataset.
*   Define and evaluate causal graphical models based on domain expertise.
*   Compare expert knowledge with causal structures learned from data using various algorithms.
*   Simulate the potential effects of interventions on key risk factors.
*   Analyze the sensitivity of diabetes risk to different factors within a causal model.
*   Facilitate a dialogue between expert knowledge and data-driven insights for refining causal understanding and supporting decision-making.

""")

st.markdown("""
## Data Description

The application utilizes a pre-processed subset of the **Behavioral Risk Factor Surveillance System (BRFSS) 2015** dataset.

*   **Source:** U.S. Centers for Disease Control and Prevention (CDC)
*   **Sample Size:** 253,680 health assessments
*   **Variables Used (22):** The following variables were selected and pre-processed (discretized) for causal analysis as described in the above mentioned paper:
    *   `Diabetes_binary`: Target variable (0: No Diabetes/Prediabetes, 1: Diabetes)
    *   `HighBP`: High Blood Pressure (0: No, 1: Yes)
    *   `HighChol`: High Cholesterol (0: No, 1: Yes)
    *   `CholCheck`: Cholesterol Check in past 5 years (0: No, 1: Yes)
    *   `BMI`: Body Mass Index (Categorized: 0: 0-24, 1: 25-39, 2: >=40)
    *   `Smoker`: Smoked at least 100 cigarettes (0: No, 1: Yes)
    *   `Stroke`: Ever had stroke (0: No, 1: Yes)
    *   `HeartDiseaseorAttack`: Coronary heart disease or myocardial infarction (0: No, 1: Yes)
    *   `PhysActivity`: Physical activity in past 30 days (0: No, 1: Yes)
    *   `Fruits`: Consume Fruit 1+ times/day (0: No, 1: Yes)
    *   `Veggies`: Consume Vegetables 1+ times/day (0: No, 1: Yes)
    *   `HvyAlcoholConsump`: Heavy alcohol consumption (0: No, 1: Yes)
    *   `AnyHealthcare`: Have any health care coverage (0: No, 1: Yes)
    *   `NoDocbcCost`: Could not see doctor due to cost (0: No, 1: Yes)
    *   `GenHlth`: General Health (Categorized: 1: Excellent, 2: Good, 3: Poor)
    *   `MentHlth`: Days mental health not good (Categorized: 0: 0-9, 1: 10-19, 2: >=20)
    *   `PhysHlth`: Days physical health not good (Categorized: 0: 0-9, 1: 10-19, 2: >=20)
    *   `DiffWalk`: Difficulty walking or climbing stairs (0: No, 1: Yes)
    *   `Sex`: (0: Female, 1: Male)
    *   `Age`: Age (Categorized: 1: 18-29, 2: 30-44, 3: 45-54, 4: 55-64, 5: 65-74, 6: >=75)
    *   `Education`: Education level (Categorized: 1: Grades 1-8, 2: Grades 9-GED, 3: College)
    *   `Income`: Income level (Categorized: 1: <$15k, 2: $15k-<$25k, 3: $25k-<$50k, 4: >=$50k)

*(Note: For further details, please refer to the above mentioned paper as well as our compiled report.)*
""")
st.markdown("---")
st.caption("Navigate using the sidebar to explore the data and causal models. It's highly recommended to cache the preloaded structures, evaluations, and results for swifter navigation.")

# Sidebar UI
st.sidebar.markdown("### Click here to cache preloaded structures, evaluations, and results:")
st.sidebar.info("(Highly Recommended)")
if st.sidebar.button("Cache"):
    from modules.bif_loader import load_all_models
    from modules.model_fitness_val import get_model_scores
    import logging
    
    # Configure logging to suppress pgmpy INFO messages
    logging.getLogger('pgmpy').setLevel(logging.WARNING)

    # Load all models and get scores
    _1 = load_all_models()
    _2 = get_model_scores()

    st.sidebar.success("Caching complete!")