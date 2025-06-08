import streamlit as st
import torch
torch.classes.__path__ = [] # adding this line to manually avoid "examining the path of torch.classes raised"; some streamlit internal error, doesn't affect app.
import pandas as pd
import plotly.express as px
import sys
import os
from modules.data_loader import load_data

# Add the project root to the Python path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis (EDA)")
st.markdown("Visualizing relationships between key variables and diabetes status based on the BRFSS 2015 dataset.")
st.markdown("---")

# Load data using the data_loader module
df = load_data()

if df.empty:
    st.error("Failed to load data. Cannot display EDA.")
else:
    # --- Variable Information Table ---
    st.subheader("Variable Information")
    
    # Define variables with their states and order
    variables = {
        'Diabetes_binary': {'states': '{0: No, 1: Yes}', 'order': 1},
        'HighBP': {'states': '{0: No, 1: Yes}', 'order': 2},
        'HighChol': {'states': '{0: No, 1: Yes}', 'order': 3},
        'BMI': {'states': '{0: 0-24, 1: 25-39, 2: ≥ 40}', 'order': 4},
        'HeartDiseaseorAttack': {'states': '{0: No, 1: Yes}', 'order': 5},
        'CholCheck': {'states': '{0: No, 1: Yes}', 'order': 6},
        'Stroke': {'states': '{0: No, 1: Yes}', 'order': 7},
        'Smoker': {'states': '{0: No, 1: Yes}', 'order': 8},
        'Fruits': {'states': '{0: No, 1: Yes}', 'order': 9},
        'Veggies': {'states': '{0: No, 1: Yes}', 'order': 10},
        'HvyAlcoholConsump': {'states': '{0: No, 1: Yes}', 'order': 11},
        'AnyHealthcare': {'states': '{0: No, 1: Yes}', 'order': 12},
        'NoDocbcCost': {'states': '{0: No, 1: Yes}', 'order': 13},
        'MentHlth': {'states': '{0: 0-9 days, 1: 10-19 days, 2: ≥ 20 days}', 'order': 14},
        'PhysHlth': {'states': '{0: 0-9 days, 1: 10-19 days, 2: ≥ 20 days}', 'order': 15},
        'DiffWalk': {'states': '{0: No, 1: Yes}', 'order': 16},
        'Sex': {'states': '{0: Female, 1: Male}', 'order': 17},
        'Age': {'states': '{1: 18-29, 2: 30-44, 3: 45-54, 4: 55-64, 5: 65-74, 6: 75 or older}', 'order': 18},
        'Income': {'states': '{1: $0-14,999, 2: $15,000-24,999, 3: $25,000-50,000, 4: ≥ $50,000}', 'order': 19},
        'Education': {'states': '{1: < 9th grade, 2: 9th grade-GED, 3: College Education}', 'order': 20},
        'GenHlth': {'states': '{1: Excellent, 2: Good, 3: Poor}', 'order': 21},
        'PhysActivity': {'states': '{0: No, 1: Yes}', 'order': 22}
    }
    
    # Calculate marginal distributions
    marginal_distributions = {}
    for var in variables:
        if var in df.columns:
            value_counts = df[var].value_counts(normalize=True).sort_index()
            marginal_distributions[var] = '{' + ', '.join([f"{pct:.0%}" for pct in value_counts.values]) + '}'
    
    # Create table data in the specified order
    table_data = []
    for var, info in sorted(variables.items(), key=lambda x: x[1]['order']):
        table_data.append({
            '#': info['order'],
            'Variable': var,
            'States': info['states'],
            'Marginal': marginal_distributions.get(var, '')
        })
    
    # Display table in a container
    with st.container(border=False):
        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
            hide_index=True,
            column_config={
                "#": st.column_config.NumberColumn(
                    width=1,
                    format="%d"
                )
            }
        )
    
    st.markdown("---")
    
    # --- Replicating Paper Figures ---

    st.subheader("Relationship Between Key Variables and Diabetes Status")
    try:
        # Create age group labels mapping
        age_labels = {
            1: '18-29',
            2: '30-44',
            3: '45-54',
            4: '55-64',
            5: '65-74',
            6: '75 or older'
        }
        
        # Calculate percentages for each age group and diabetes status
        age_diabetes_counts = df.groupby(['Age', 'Diabetes_binary'], observed=False).size().unstack(fill_value=0)
        age_diabetes_pct = age_diabetes_counts.div(age_diabetes_counts.sum(axis=0), axis=1) * 100
        
        # Create a new dataframe for plotting
        plot_data = []
        for age in age_diabetes_pct.index:
            for diabetes_status in [0, 1]:
                plot_data.append({
                    'Age Group': age_labels[age],
                    'Percentage': age_diabetes_pct.loc[age, diabetes_status],
                    'Patient Type': 'Non-Diabetic' if diabetes_status == 0 else 'Diabetic'
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create the plot
        fig_age = px.bar(
            plot_df,
            x='Age Group',
            y='Percentage',
            color='Patient Type',
            barmode='group',
            title='Percentage of Age Groups Among Patients',
            category_orders={'Age Group': list(age_labels.values())}
        )
        
        # Update layout
        fig_age.update_layout(
            xaxis_title='Age Group',
            yaxis_title='Percentage',
            legend_title='Patient Type'
        )
        
        st.plotly_chart(fig_age, use_container_width=True)
        st.caption("Age distribution of diabetic and non-diabetic individuals.")
    except Exception as e:
        st.warning(f"Could not generate Age plot: {e}")

    try:
        # Create income range labels mapping
        income_labels = {
            1: '$0-14,999',
            2: '$15,000-24,999',
            3: '$25,000-50,000',
            4: '≥$50,000'
        }
        
        # Calculate percentages for each income group and diabetes status
        income_diabetes_counts = df.groupby(['Income', 'Diabetes_binary'], observed=False).size().unstack(fill_value=0)
        income_diabetes_pct = income_diabetes_counts.div(income_diabetes_counts.sum(axis=0), axis=1) * 100
        
        # Create a new dataframe for plotting
        plot_data = []
        for income in income_diabetes_pct.index:
            for diabetes_status in [0, 1]:
                plot_data.append({
                    'Income Group': income_labels[income],
                    'Percentage': income_diabetes_pct.loc[income, diabetes_status],
                    'Patient Type': 'Non-Diabetic' if diabetes_status == 0 else 'Diabetic'
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create the plot
        fig_income = px.bar(
            plot_df,
            x='Income Group',
            y='Percentage',
            color='Patient Type',
            barmode='group',
            title='Percentage of Income Groups Among Patients',
            category_orders={'Income Group': ['$0-14,999', '$15,000-24,999', '$25,000-50,000', '≥$50,000']}
        )
        
        # Update layout
        fig_income.update_layout(
            xaxis_title='Income Group',
            yaxis_title='Percentage',
            legend_title='Patient Type'
        )
        
        st.plotly_chart(fig_income, use_container_width=True)
        st.caption("Annual income distribution of diabetic and non-diabetic individuals.")
    except Exception as e:
        st.warning(f"Could not generate Income plot: {e}")

    try:
        # Create general health labels mapping
        genhlth_labels = {
            1: 'Excellent',
            2: 'Good',
            3: 'Poor'
        }
        
        # Calculate percentages for each general health category and diabetes status
        genhlth_diabetes_counts = df.groupby(['GenHlth', 'Diabetes_binary'], observed=False).size().unstack(fill_value=0)
        genhlth_diabetes_pct = genhlth_diabetes_counts.div(genhlth_diabetes_counts.sum(axis=0), axis=1) * 100
        
        # Create a new dataframe for plotting
        plot_data = []
        for genhlth in genhlth_diabetes_pct.index:
            for diabetes_status in [0, 1]:
                plot_data.append({
                    'General Health': genhlth_labels[genhlth],
                    'Percentage': genhlth_diabetes_pct.loc[genhlth, diabetes_status],
                    'Patient Type': 'Non-Diabetic' if diabetes_status == 0 else 'Diabetic'
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create the plot
        fig_genhlth = px.bar(
            plot_df,
            x='General Health',
            y='Percentage',
            color='Patient Type',
            barmode='group',
            title='Percentage of General Health Categories Among Patients',
            category_orders={'General Health': ['Excellent', 'Good', 'Poor']}
        )
        
        # Update layout
        fig_genhlth.update_layout(
            xaxis_title='General Health',
            yaxis_title='Percentage',
            legend_title='Patient Type'
        )
        
        st.plotly_chart(fig_genhlth, use_container_width=True)
        st.caption("General Health distribution of diabetic and non-diabetic individuals.")
    except Exception as e:
        st.warning(f"Could not generate General Health plot: {e}")

    try:
        # Create BMI category labels mapping
        bmi_labels = {
            0: '0-24',
            1: '25-39',
            2: '≥40'
        }
        
        # Calculate percentages for each BMI category and diabetes status
        bmi_diabetes_counts = df.groupby(['BMI', 'Diabetes_binary'], observed=False).size().unstack(fill_value=0)
        bmi_diabetes_pct = bmi_diabetes_counts.div(bmi_diabetes_counts.sum(axis=0), axis=1) * 100
        
        # Create a new dataframe for plotting
        plot_data = []
        for bmi in bmi_diabetes_pct.index:
            for diabetes_status in [0, 1]:
                plot_data.append({
                    'BMI Categories': bmi_labels[bmi],
                    'Percentage': bmi_diabetes_pct.loc[bmi, diabetes_status],
                    'Patient Type': 'Non-Diabetic' if diabetes_status == 0 else 'Diabetic'
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create the plot
        fig_bmi = px.bar(
            plot_df,
            x='BMI Categories',
            y='Percentage',
            color='Patient Type',
            barmode='group',
            title='BMI Categories of Patients',
            category_orders={'BMI Categories': ['0-24', '25-39', '≥40']}
        )
        
        # Update layout
        fig_bmi.update_layout(
            xaxis_title='BMI Categories',
            yaxis_title='Percentage',
            legend_title='Patient Type'
        )
        
        st.plotly_chart(fig_bmi, use_container_width=True)
        st.caption("BMI distribution of diabetic and non-diabetic individuals.")
    except Exception as e:
        st.warning(f"Could not generate BMI plot: {e}")

    try:
        # Create cholesterol status labels mapping
        chol_labels = {
            0: 'No',
            1: 'Yes'
        }
        
        # Calculate percentages for each cholesterol status and diabetes status
        chol_diabetes_counts = df.groupby(['HighChol', 'Diabetes_binary'], observed=False).size().unstack(fill_value=0)
        chol_diabetes_pct = chol_diabetes_counts.div(chol_diabetes_counts.sum(axis=0), axis=1) * 100
        
        # Create a new dataframe for plotting
        plot_data = []
        for chol in chol_diabetes_pct.index:
            for diabetes_status in [0, 1]:
                plot_data.append({
                    'High Cholesterol': chol_labels[chol],
                    'Percentage': chol_diabetes_pct.loc[chol, diabetes_status],
                    'Patient Type': 'Non-Diabetic' if diabetes_status == 0 else 'Diabetic'
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create the plot
        fig_chol = px.bar(
            plot_df,
            x='High Cholesterol',
            y='Percentage',
            color='Patient Type',
            barmode='group',
            title='Percentage of High Cholesterol Among Patients',
            category_orders={'High Cholesterol': ['No', 'Yes']}
        )
        
        # Update layout
        fig_chol.update_layout(
            xaxis_title='High Cholesterol',
            yaxis_title='Percentage',
            legend_title='Patient Type'
        )
        
        st.plotly_chart(fig_chol, use_container_width=True)
        st.caption("High Cholesterol distribution of diabetic and non-diabetic individuals.")
    except Exception as e:
        st.warning(f"Could not generate High Cholesterol plot: {e}")

    try:
        # Create blood pressure status labels mapping
        bp_labels = {
            0: 'No',
            1: 'Yes'
        }
        
        # Calculate percentages for each blood pressure status and diabetes status
        bp_diabetes_counts = df.groupby(['HighBP', 'Diabetes_binary'], observed=False).size().unstack(fill_value=0)
        bp_diabetes_pct = bp_diabetes_counts.div(bp_diabetes_counts.sum(axis=0), axis=1) * 100
        
        # Create a new dataframe for plotting
        plot_data = []
        for bp in bp_diabetes_pct.index:
            for diabetes_status in [0, 1]:
                plot_data.append({
                    'High Blood Pressure': bp_labels[bp],
                    'Percentage': bp_diabetes_pct.loc[bp, diabetes_status],
                    'Patient Type': 'Non-Diabetic' if diabetes_status == 0 else 'Diabetic'
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create the plot
        fig_bp = px.bar(
            plot_df,
            x='High Blood Pressure',
            y='Percentage',
            color='Patient Type',
            barmode='group',
            title='Percentage of High Blood Pressure Among Patients',
            category_orders={'High Blood Pressure': ['No', 'Yes']}
        )
        
        # Update layout
        fig_bp.update_layout(
            xaxis_title='High Blood Pressure',
            yaxis_title='Percentage',
            legend_title='Patient Type'
        )
        
        st.plotly_chart(fig_bp, use_container_width=True)
        st.caption("High Blood Pressure distribution of diabetic and non-diabetic individuals.")
    except Exception as e:
        st.warning(f"Could not generate High Blood Pressure plot: {e}")

    # --- Interactive Variable Analysis ---
    st.markdown("---")
    st.subheader("Interactive Variable Distribution")

    # Create a selectbox for variable selection
    selected_var = st.selectbox(
        "Select a variable to view its distribution by diabetes status:",
        options=list(variables.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        index=list(variables.keys()).index('Smoker')  # Set Smoker as default
    )

    if selected_var:
        try:
            # Get the states mapping from the variables dictionary
            states_str = variables[selected_var]['states']
            # Convert the states string to a dictionary
            states_dict = eval(states_str.replace('{', '{"').replace(':', '":"').replace(', ', '", "').replace('}', '"}'))
            
            # Calculate percentages for the selected variable and diabetes status
            var_diabetes_counts = df.groupby([selected_var, 'Diabetes_binary'], observed=False).size().unstack(fill_value=0)
            var_diabetes_pct = var_diabetes_counts.div(var_diabetes_counts.sum(axis=0), axis=1) * 100
            
            # Create a new dataframe for plotting
            plot_data = []
            for var_value in var_diabetes_pct.index:
                for diabetes_status in [0, 1]:
                    plot_data.append({
                        'Category': states_dict[str(var_value)],
                        'Percentage': var_diabetes_pct.loc[var_value, diabetes_status],
                        'Patient Type': 'Non-Diabetic' if diabetes_status == 0 else 'Diabetic'
                    })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Create appropriate titles
            x_axis_title = selected_var.replace('_', ' ').title()
            graph_title = f"Percentage of {x_axis_title} Categories Among Patients"
            
            # Create the plot
            fig_custom = px.bar(
                plot_df,
                x='Category',
                y='Percentage',
                color='Patient Type',
                barmode='group',
                title=graph_title,
                category_orders={'Category': list(states_dict.values())}
            )
            
            # Update layout
            fig_custom.update_layout(
                xaxis_title=x_axis_title,
                yaxis_title='Percentage',
                legend_title='Patient Type'
            )
            
            st.plotly_chart(fig_custom, use_container_width=True)
            st.caption(f"{x_axis_title} distribution of diabetic and non-diabetic individuals.")
            
        except Exception as e:
            st.warning(f"Could not generate plot for {selected_var}: {e}")
st.markdown("---")