import streamlit as st
import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diabetescategorised.csv")

@st.cache_data # Cache the data loading to improve performance
def load_data():
    """Loads the preprocessed BRFSS data from the CSV file."""
    try:
        df = pd.read_csv(DATA_PATH)
        categorical_cols = df.columns
        for col in categorical_cols:
            df[col] = pd.Categorical(df[col])
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {DATA_PATH}")
        # Return an empty dataframe or handle appropriately
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()

# Example usage (optional, for testing module directly)
if __name__ == "__main__":
    df_loaded = load_data()
    if not df_loaded.empty:
        print("Data loaded successfully:")
        print(df_loaded.head())
        print("\nData Info:")
        print(df_loaded.info())
    else:
        print("Failed to load data.")

