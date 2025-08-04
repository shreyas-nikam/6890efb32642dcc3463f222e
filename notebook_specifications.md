
## Credit Risk Forecasting Lab: Jupyter Notebook Specification

### 1. Notebook Overview

**Learning Goals:**

This notebook aims to provide a hands-on experience in building macroeconomic time series models for credit risk forecasting. Users will learn to:

- Integrate and prepare credit-risk and macroeconomic data.
- Identify structural breaks and regimes in time series data.
- Transform data to achieve stationarity.
- Build and evaluate econometric models (ARDL, VAR, ARIMAX) for credit risk forecasting.
- Implement scenario forecasting and stress testing.
- Effectively communicate results through visualizations and narratives.

**Expected Outcomes:**

Upon completion of this notebook, users should be able to:

- Understand the relationship between macroeconomic factors and credit risk metrics (PD and LGD).
- Apply data cleaning and transformation techniques to time series data.
- Develop and evaluate econometric models for credit risk forecasting.
- Implement scenario forecasting and stress testing methodologies.
- Interpret and communicate the impact of macroeconomic scenarios on credit risk.

### 2. Mathematical and Theoretical Foundations

This section will provide a concise overview of the relevant mathematical and statistical concepts.

**2.1 Probability of Default (PD) and Loss Given Default (LGD):**

- **Definition:**  $PD$ is the probability that a borrower will default on their debt obligations within a specified time horizon. $LGD$ is the percentage of loss expected if a default occurs.
- **Real-world Application:**  $PD$ and $LGD$ are crucial inputs for credit risk management, regulatory capital calculations, and loan pricing.
- **Formula:** Expected Loss (EL) can be defined as $EL = PD * LGD * EAD$, where EAD is Exposure at Default.

**2.2 Stationarity:**

- **Definition:** A time series is stationary if its statistical properties (mean, variance, autocorrelation) do not change over time.
- **Real-world Application:** Many time series models require stationarity for accurate forecasting.
- **Testing:** Stationarity can be verified using Augmented Dickey-Fuller (ADF) and Phillips-Perron tests.

**2.3 Autoregressive Distributed Lag (ARDL) Model:**

- **Definition:** An ARDL model is a regression model that includes lagged values of both the dependent variable and independent variables.

- **Formula:** The general form of an ARDL(p, q, r) model is:

$$y_t = \alpha + \sum_{i=1}^{p} \phi_i y_{t-i} + \sum_{j=1}^{q} \beta_j x_{1,t-j} + \sum_{k=1}^{r} \gamma_k x_{2,t-k} + \epsilon_t$$

Where:
    - $y_t$ is the dependent variable at time $t$.
    - $x_{1,t}$ and $x_{2,t}$ are independent variables at time $t$.
    - $p$, $q$, and $r$ are the lag orders for the dependent and independent variables, respectively.
    - $\phi_i$, $\beta_j$, and $\gamma_k$ are coefficients.
    - $\epsilon_t$ is the error term.
- **Real-world Application:** ARDL models are useful for capturing both the persistence of the dependent variable and the lagged effects of independent variables.  An example, as mentioned in the prompt, could be:

$$ \text{DefaultRate}_t = \alpha + \sum_{i=1}^{p} \phi_i \text{DefaultRate}_{t-i} + \sum_{j=1}^{q} \beta_j \text{Unemployment}_{t-j} + \sum_{k=1}^{r} \gamma_k \text{GDPGrowth}_{t-k} + \epsilon_t$$

**2.4 Vector Autoregression (VAR) Model:**

- **Definition:** A VAR model is a multivariate time series model that captures the interdependencies between multiple variables.
- **Formula:**
For a two-variable VAR(p) model:
$$y_t = c + A_1 y_{t-1} + A_2 y_{t-2} + \dots + A_p y_{t-p} + \epsilon_t$$
where $y_t = \begin{bmatrix} y_{1t} \\ y_{2t} \end{bmatrix}$ is a vector of two time series variables, $c$ is a constant vector, $A_i$ are coefficient matrices, and $\epsilon_t$ is a vector of error terms.
- **Real-world Application:** VAR models are useful for jointly modeling multiple related time series, such as default rates, GDP growth, and unemployment.

**2.5 ARIMAX Model:**

- **Definition:**  An ARIMAX model extends the ARIMA model by including exogenous variables.
- **Formula:** The ARIMAX model can be expressed as:
$(1 - \sum_{i=1}^{p} \phi_i L^i)(1-L)^d y_t = (1 + \sum_{i=1}^{q} \theta_i L^i) (\sum_{j=1}^{k} \beta_j x_{jt} + \epsilon_t)$
Where:
$y_t$ is the time series being forecasted.
$x_{jt}$ are the exogenous variables.
$L$ is the lag operator.
$p, d, q$ are the orders of the autoregressive, integrated, and moving average parts of the model.
$\phi_i, \theta_i, \beta_j$ are the model coefficients.
$\epsilon_t$ is the error term.
- **Real-world Application:** ARIMAX models are useful when the time series is influenced by external factors, allowing for more accurate forecasting.

**2.6 Variance Inflation Factor (VIF):**

- **Definition:** VIF quantifies the severity of multicollinearity in a multiple regression model. It measures how much the variance of an estimated regression coefficient increases if your predictors are correlated.
- **Formula:** $VIF_i = \frac{1}{1 - R_i^2}$ , where $R_i^2$ is the R-squared value from regressing the $i$-th predictor on all other predictors in the model.
- **Real-world Application:** High VIF values (typically > 5) indicate strong multicollinearity, which can lead to unstable and unreliable regression coefficients.

### 3. Code Requirements

**3.1 Expected Libraries:**

-   `pandas`: For data manipulation and analysis (ingestion, cleaning, transformation, and storage).
-   `numpy`: For numerical computations.
-   `matplotlib` or `seaborn`: For creating visualizations (time series plots, correlation heatmaps, forecast outputs).
-   `statsmodels`: For econometric modeling (ARDL, VAR, ARIMAX), stationarity tests (ADF, Phillips-Perron), and VIF calculation.
-   `scikit-learn`: For potential seasonal decomposition or detrending.
-   `pmdarima`: For easier ARIMA model selection.

**3.2 Input/Output Expectations:**

-   **Input:**
    -   Synthetic datasets containing quarterly credit-risk series (default rates by segment, average LGDs) and macroeconomic drivers (Real GDP growth, oil prices, inflation, unemployment rate, interest rates, stock index). Data will come from CSV. User will also be able to enter custom data.
    -   User-defined macroeconomic scenarios (grid or JSON format).
-   **Output:**
    -   DataFrames containing cleaned and transformed data.
    -   Model summaries and diagnostic statistics.
    -   Point forecasts of credit risk metrics (PD and LGD).
    -   Visualizations (time series plots, correlation heatmaps, forecast outputs).
    -   CSV/Excel files containing point forecasts.
    -   Transformation log.

**3.3 Algorithms/Functions to be Implemented:**

1.  **Data Ingestion and Cleaning Functions:**
    -   Function to import credit-risk and macroeconomic data from CSV files.
    -   Function to perform data quality checks (missing values, outliers).
    -   Function to align timelines and manage lags/missing values.

2.  **Data Transformation Functions:**
    -   Function to apply YoY %/QoQ % changes.
    -   Function to apply log-differences.
    -   Function to perform seasonal adjustment.
    -   Function to normalize data (e.g., real terms, per-capita or GDP-scaled metrics).

3.  **Stationarity Testing Functions:**
    -   Function to perform ADF test.
    -   Function to perform Phillips-Perron test.

4.  **Correlation and Multicollinearity Check Functions:**
    -   Function to compute pairwise correlations.
    -   Function to compute VIF.

5.  **Model Building Functions:**
    -   Function to fit ARDL models.
    -   Function to fit VAR models.
    -   Function to fit ARIMAX models.
    -   Function to calculate goodness-of-fit measures (Adjusted R², AIC/BIC).

6.  **Scenario Forecasting Functions:**
    -   Function to accept user-defined macroeconomic scenarios.
    -   Function to implement recursive forecasting logic.

7.  **Visualization Functions:**
    -   Function to generate time-series plots.
    -   Function to generate correlation heatmaps.
    -   Function to generate forecast outputs (term structure plots, stress multipliers).

**3.4 Visualizations:**

-   Exploratory time-series plots: macro variable vs corresponding credit metric with shaded break periods.
-   Correlation & lag visuals: heat-map of pairwise correlations and cross-correlation bar plots.
-   Forecast outputs: line chart of historical + baseline + stress scenarios (plus confidence bands if produced) and bar chart/table of stress multipliers.

### 4. Additional Notes or Instructions

-   **Assumptions:**
    -   Synthetic data is representative of real-world credit risk and macroeconomic relationships.
    -   The chosen macroeconomic variables are relevant drivers of credit risk.
    -   Stationarity is a necessary condition for accurate forecasting.
-   **Constraints:**
    -   The models are limited to ARDL, VAR, and ARIMAX frameworks.
    -   The data is limited to quarterly frequency.
-   **Customization Instructions:**
    -   Users should be able to select the jurisdiction and segment for analysis.
    -   Users should be able to specify the lag structures for the models.
    -   Users should be able to define custom macroeconomic scenarios.
    -   Users should be able to save the point forecasts and plots to CSV/Excel files.
- **Reproducibility features:** fixed random seed, printed library versions, and parameterised jurisdiction/segment selectors.
- **Transformation Log**: Automatically track variable transformations and stationarity results.
- **Correlation & Multicollinearity Checks**: Compute correlations and VIF, with warnings when VIF > 5.
- **Beginner-friendly commentary**: inline explanations linking economic reasoning to each code step.

