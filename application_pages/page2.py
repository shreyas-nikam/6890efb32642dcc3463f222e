import streamlit as st
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import adfuller
import numpy as np
import warnings

def run_page2():
    st.header("2. Data Pre-processing for Stationarity")
    st.markdown(r"""
    This step prepares the data for ARIMAX by checking **stationarity** and applying minimal transformations.

    You will:
    - Run **ADF** and **KPSS** on the original series.
    - Difference the default rate and re-test.
    - Store the transformed dataset for modeling on the next page.

    How to read the tests:
    - **ADF** p ≤ 0.05 → stationary | p > 0.05 → non-stationary.
    - **KPSS** p ≤ 0.05 → non-stationary |  p > 0.05 → stationary. 
    - Use both to decide if differencing is needed.
    """)

    def run_unit_root_tests(series, name):
        """
        Performs Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests
        for stationarity on a given time series.
        """
        adf_result = adfuller(series.dropna(), autolag='AIC')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(series.dropna(), regression='c', nlags='auto')

        results_df = pd.DataFrame({
            "Test": ["ADF", "KPSS"],
            "Statistic": [round(adf_result[0], 4), round(kpss_result[0], 4)],
            "P-value": [round(adf_result[1], 4), round(kpss_result[1], 4)],
            "Stationary?": [
                "Yes" if adf_result[1] <= 0.05 else "No",
                "Yes" if kpss_result[1] > 0.05 else "No"
            ]
        })

        st.subheader(f"Unit Root Tests for: {name}")
        st.dataframe(results_df)
        return adf_result, kpss_result

    st.subheader("Transformations for Stationarity:")
    st.markdown(r"""
    *   **Year-over-Year (YoY) Percentage Change**:
        $$ Y_{t}^{YoY} = \frac{Y_t - Y_{t-4}}{Y_{t-4}} \times 100 $$
    *   **Log-Differencing**:
        $$ \Delta \ln(Y_t) = \ln(Y_t) - \ln(Y_{t-1}) $$
    """)

    if st.button("Apply Transformations and Test Stationarity"):
        # Load the synthetic data (same as in page1.py)
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
            'GDP_Growth_YoY(%)': gdp_growth,
            'Unemployment(%)': unemployment
        })
        df_synthetic.set_index('Quarter', inplace=True)
        df_synthetic.index.freq = 'QS-JAN'

        df_transformed = df_synthetic.copy()
        df_transformed['Default_Rate_Diff'] = df_transformed['Segment A Default Rate'].diff(1)
        df_transformed['Unemployment_Diff'] = df_transformed['Unemployment(%)'].diff(1)
        df_transformed = df_transformed.drop(columns=['Segment A Default Rate'])
        df_transformed = df_transformed.dropna()
        df_transformed.index = pd.to_datetime(df_transformed.index)
        df_transformed.index.freq = 'QS-JAN'
        st.markdown("""
        Below are the stationarity test results for each variable before (left) and after (right) transformation. The transformation’s goal is to make each series suitable for ARIMAX by achieving stationarity. A series is considered ready if **ADF** shows p ≤ 0.05 and **KPSS** shows p > 0.05.
        """)

        col1, col2 = st.columns(2)

        
        with col1:
            st.subheader("Original time series:")
            st.divider()
            run_unit_root_tests(df_synthetic['Segment A Default Rate'], 'Segment A Default Rate')
            run_unit_root_tests(df_synthetic['GDP_Growth_YoY(%)'], 'GDP_Growth_YoY(%)')
            run_unit_root_tests(df_synthetic['Unemployment(%)'], 'Unemployment(%)')

        with col2:
            st.subheader("Transformed time series:")
            st.divider()
            run_unit_root_tests(df_transformed['Default_Rate_Diff'], 'Default_Rate_Diff')
            run_unit_root_tests(df_transformed['GDP_Growth_YoY(%)'], 'GDP_Growth_YoY(%) ')
            run_unit_root_tests(df_transformed['Unemployment_Diff'].dropna(), 'Unemployment(%)')

        st.session_state['df_transformed'] = df_transformed  # Store for next step

if __name__ == "__main__":
    run_page2()
