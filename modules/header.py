import streamlit as st

def show_header():
    # Create a container for the header to ensure it's at the top
    with st.container():
        # Use a 1:4 ratio to give more space for the title
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("assets/logo.png", width=300)  # Reduced width to better fit the layout
        with col2:
            st.markdown("<br>", unsafe_allow_html=True) 
            st.title("DiaCausal: A Diabetes Decision Support")
            # Add some spacing to push content down
            st.markdown("<br>", unsafe_allow_html=True) 