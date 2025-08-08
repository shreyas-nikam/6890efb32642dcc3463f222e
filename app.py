
import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we will explore macro-economic models and their application to credit risk forecasting. We will use a synthetic dataset to estimate and diagnose an ARIMAX model for Segment A default rates, incorporating key macroeconomic variables.
formulae, explanations, tables, etc.
""")
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Data Loading & Exploration", "Data Pre-processing (Stationarity)", "ARIMAX Model Estimation & Diagnostics", "Model Persistence"])
if page == "Data Loading & Exploration":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Data Pre-processing (Stationarity)":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "ARIMAX Model Estimation & Diagnostics":
    from application_pages.page3 import run_page3
    run_page3()
elif page == "Model Persistence":
    from application_pages.page4 import run_page4
    run_page4()
# Your code ends
