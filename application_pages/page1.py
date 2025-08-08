
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_page1():
    st.header("1. Data Loading and Initial Exploration")
    st.markdown("This section loads synthetic time series data mimicking Taiwan credit risk and macroeconomic data. It displays initial data insights and visualizations.")

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
    st.write(f"Dataset shape: {df_synthetic.shape}")
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

if __name__ == "__main__":
    run_page1()
