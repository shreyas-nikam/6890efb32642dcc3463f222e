
import streamlit as st
import pandas as pd
import statsmodels.stats.diagnostic as smd

def run_page2():
    st.header("2. Data Pre-processing for Stationarity")
    st.markdown("This section defines a helper function for unit root tests and applies differencing to the target variable to achieve stationarity, then verifies it.")

    def run_unit_root_tests(series, name):
        """
        Performs Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests
        for stationarity on a given time series.
        """
        st.write(f"\n--- Unit Root Tests for: **{name}** ---")

        # ADF Test
        # Null Hypothesis ($H_0$): The series has a unit root (non-stationary).
        # If p-value <= 0.05, reject $H_0$ (series is stationary).
        adf_result = smd.adfuller(series.dropna(), autolag='AIC')
        st.write("Augmented Dickey-Fuller Test:")
        st.write(f"  ADF Statistic: {adf_result[0]:.4f}")
        st.write(f"  P-value: {adf_result[1]:.4f}")
        st.write(f"  Critical Values (1%, 5%, 10%): {adf_result[4]}")
        st.write(f"  Result: {'**Stationary**' if adf_result[1] <= 0.05 else '**Non-Stationary**'} (based on ADF)")

        # KPSS Test
        # Null Hypothesis ($H_0$): The series is stationary around a deterministic trend (trend-stationary).
        # If p-value <= 0.05, reject $H_0$ (series is non-stationary).
        try:
            kpss_result = smd.kpss(series.dropna(), regression='c', nlags='auto')
            st.write("\nKPSS Test:")
            st.write(f"  KPSS Statistic: {kpss_result[0]:.4f}")
            st.write(f"  P-value: {kpss_result[1]:.4f}")
            st.write(f"  Critical Values (10%, 5%, 2.5%, 1%): {kpss_result[3]}")
            st.write(f"  Result: {'**Non-Stationary**' if kpss_result[1] <= 0.05 else '**Stationary**'} (based on KPSS)")
        except Exception as e:
            st.error(f"\nKPSS Test failed: {e}")
        
        return adf_result, kpss_result if 'kpss_result' in locals() else None

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
            'GDP_Growth_YoY_%': gdp_growth,
            'Unemployment_%': unemployment
        })
        df_synthetic.set_index('Quarter', inplace=True)
        df_synthetic.index.freq = 'QS-JAN'

        st.subheader("Testing stationarity of original time series:")
        run_unit_root_tests(df_synthetic['Segment A Default Rate'], 'Segment A Default Rate')
        run_unit_root_tests(df_synthetic['GDP_Growth_YoY_%'], 'GDP_Growth_YoY_%')
        run_unit_root_tests(df_synthetic['Unemployment_%'], 'Unemployment_%')

        df_transformed = df_synthetic.copy()
        df_transformed['Default_Rate_Diff'] = df_transformed['Segment A Default Rate'].diff(1)
        df_transformed = df_transformed.drop(columns=['Segment A Default Rate'])
        df_transformed = df_transformed.dropna()
        df_transformed.index = pd.to_datetime(df_transformed.index)
        df_transformed.index.freq = 'QS-JAN'

        st.subheader("Testing stationarity of transformed time series:")
        run_unit_root_tests(df_transformed['Default_Rate_Diff'], 'Default_Rate_Diff')
        run_unit_root_tests(df_transformed['GDP_Growth_YoY_%'], 'GDP_Growth_YoY_% (Transformed)')

        st.session_state['df_transformed'] = df_transformed  # Store for next step

if __name__ == "__main__":
    run_page2()
