
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_page1():

    st.markdown(r"""

    This lab is designed to help you understand how macroeconomic trends influence credit risk and how to capture those relationships in a predictive time series model.  
    Using synthetic quarterly data inspired by Taiwan’s economy, you will follow a guided process to prepare, build, and evaluate an ARIMAX model.

    ## Learning Objectives
    - Understand how GDP growth, unemployment, and other indicators impact default rates.
    - Apply statistical tests (ADF, KPSS) to check for stationarity and perform necessary transformations.
    - Build an ARIMAX model that incorporates macroeconomic variables as exogenous inputs.
    - Compare model configurations using AIC/BIC and interpret residual diagnostics.
    - Save your best-performing model for reuse or deployment.

    ## Lab Structure
    1. **Data Loading & Exploration** – Load the synthetic dataset, review summary statistics, and visualize economic and credit risk trends.
    2. **Data Pre-processing** – Apply unit root tests and transform non-stationary series to meet modeling requirements.
    3. **Model Estimation & Diagnostics** – Train ARIMAX, assess fit with statistical criteria, and evaluate residual behavior.
    4. **Model Persistence** – Save the trained model as a `.pkl` file for future forecasting.

    ## Why This Lab
    Macroeconomic conditions strongly influence credit behavior. This lab gives you hands-on experience in quantifying those effects and integrating them into a forecasting model that can support better risk assessment and planning.

    ---
    """)
    
    st.header("1. Data Loading and Initial Exploration")
    st.markdown(r"""
    In this step, we create and explore a synthetic quarterly dataset representing Taiwan’s credit risk and key macroeconomic indicators.  
    The data includes default rates (target variable), GDP growth, and unemployment, with patterns that mimic real economic cycles, including a simulated recession.  
    You will review the dataset’s structure, basic statistics, and visualizations to understand how these variables behave over time and relate to each other.
    """)

    RANDOM_SEED = 42
    start_date = '2015-01-01'
    quarters = pd.date_range(start=start_date, periods=40, freq='QS-JAN')

    np.random.seed(RANDOM_SEED)
    default_rate = 1.5 + 0.05 * np.arange(len(quarters)) + np.random.normal(0, 0.2, len(quarters))
    recession_start = 20
    recession_end = 24
    default_rate[recession_start:recession_end] += np.random.normal(1.0, 0.3, recession_end - recession_start)

    gdp_growth = 2.5 + 1.0 * np.sin(np.arange(len(quarters)) * 2 * np.pi / 8) + np.random.normal(0, 0.5, len(quarters))
    gdp_growth[recession_start:recession_end] = np.random.normal(-2.0, 1.0, recession_end - recession_start)

    unemployment = 5.0 - 0.5 * gdp_growth + np.random.normal(0, 0.3, len(quarters))
    unemployment = np.clip(unemployment, 2.0, 12.0)

    df_synthetic = pd.DataFrame({
        'Quarter': quarters,
        'Segment A Default Rate': default_rate,
        'GDP_Growth_YoY_%': gdp_growth,
        'Unemployment_%': unemployment
    })
    df_synthetic.set_index('Quarter', inplace=True)
    df_synthetic.index.freq = 'QS-JAN'

    st.subheader("Synthetic Dataset Overview:")
    st.dataframe(df_synthetic.head())
    st.write("Basic statistics:")
    st.dataframe(df_synthetic.describe())

    # Visualization of synthetic time series data
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes[0].plot(df_synthetic.index, df_synthetic['Segment A Default Rate'], linewidth=2, color='red', label='Segment A Default Rate')
    axes[0].set_title('Segment A Default Rate Over Time', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Default Rate (%)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df_synthetic.index, df_synthetic['GDP_Growth_YoY_%'], linewidth=2, color='blue', label='GDP Growth YoY%')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('GDP Growth Year-over-Year (%)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('GDP Growth (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(df_synthetic.index, df_synthetic['Segment A Default Rate'], linewidth=2, color='red', label='Default Rate')
    line2 = ax3_twin.plot(df_synthetic.index, df_synthetic['GDP_Growth_YoY_%'], linewidth=2, color='blue', label='GDP Growth')
    ax3.set_title('Default Rate vs GDP Growth Over Time', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Default Rate (%)', color='red')
    ax3_twin.set_ylabel('GDP Growth (%)', color='blue')
    ax3.set_xlabel('Quarter')
    ax3.grid(True, alpha=0.3)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # Prevent duplicate plots

    st.markdown("""
    **Graph Interpretation:**  
    - The default rate generally trends upward, with a clear spike during the simulated recession period.  
    - GDP growth follows cyclical patterns and drops sharply during the recession, while unemployment moves inversely to GDP.  
    - The combined plot shows the negative relationship between GDP growth and default rates.
    """)


if __name__ == "__main__":
    run_page1()
