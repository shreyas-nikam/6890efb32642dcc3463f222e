# QuLab: Macroeconomic ARIMAX Model for Credit Risk Forecasting

## Project Title and Description

**QuLab** is an interactive Streamlit application designed as a lab project to explore the application of macroeconomic models to credit risk forecasting. Using a synthetic dataset that mimics Taiwan's economic indicators and credit default rates, the application guides users through the process of estimating and diagnosing an ARIMAX (AutoRegressive Integrated Moving Average with eXogenous regressors) model.

The primary goal of this lab is to demonstrate how macroeconomic variables can be integrated into time series models to predict credit default rates, providing a practical understanding of econometric modeling in a financial context.

## Features

This application provides a step-by-step workflow, organized into distinct sections accessible via the sidebar navigation:

*   **1. Data Loading & Exploration:**
    *   Generates a synthetic time series dataset for Segment A Default Rates, GDP Growth, and Unemployment.
    *   Displays the head, shape, and basic descriptive statistics of the generated data.
    *   Visualizes the individual time series trends and the relationship between default rates and GDP growth.

*   **2. Data Pre-processing (Stationarity):**
    *   Explains key transformations like Year-over-Year (YoY) percentage change and Log-Differencing.
    *   Implements and performs Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests to assess the stationarity of original and transformed series.
    *   Guides the user to apply appropriate differencing to achieve stationarity for the target variable (`Segment A Default Rate`).

*   **3. ARIMAX Model Estimation & Diagnostics:**
    *   Presents the mathematical formulation of the ARIMAX(p,d,q) model.
    *   Explains model selection criteria (AIC and BIC) and residual diagnostics (Ljung-Box test).
    *   Allows users to specify the `p`, `d`, and `q` orders for the ARIMAX model.
    *   Trains the ARIMAX model using the transformed default rate and selected macroeconomic variables (e.g., GDP Growth).
    *   Displays a comprehensive model summary including coefficients, standard errors, p-values, AIC, and BIC.
    *   Provides residual diagnostics plots (ACF and PACF) and Ljung-Box test results to check for remaining autocorrelation.
    *   Offers a comparison table for different trained models based on their AIC/BIC values.

*   **4. Model Persistence:**
    *   Enables users to download the best-performing fitted ARIMAX model (as a `.pkl` file) for future use, such as deployment or further analysis.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.7+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
    *(Replace `your-username/your-repository-name` with the actual repository URL)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file in the root directory of your project with the following content:

    ```
    streamlit
    pandas
    numpy
    matplotlib
    statsmodels
    ```

    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application, navigate to the root directory of your cloned project in the terminal and execute:

```bash
streamlit run app.py
```

This command will open the application in your default web browser.

### Basic Usage Flow:

1.  **Navigate the Sidebar:** Use the `Navigation` select box in the left sidebar to switch between different lab sections.
2.  **Data Loading & Exploration:** Start here to understand the synthetic dataset used for the lab. Review the tables and plots.
3.  **Data Pre-processing (Stationarity):** Click the "Apply Transformations and Test Stationarity" button to see the unit root test results for the original and transformed series. This step is crucial for preparing data for ARIMAX modeling.
4.  **ARIMAX Model Estimation & Diagnostics:**
    *   Enter your desired `p`, `d`, and `q` orders for the ARIMAX model.
    *   Click "Train Selected ARIMAX Model" to estimate the model and view its summary, residual diagnostics plots, and Ljung-Box test results.
    *   Experiment with different `(p,d,q)` orders to observe changes in AIC/BIC and residual properties.
5.  **Model Persistence:** Once a model is trained, you can download it as a `.pkl` file for later use.

## Project Structure

The project is organized into a modular structure to enhance readability and maintainability:

```
├── app.py
├── requirements.txt
└── application_pages/
    ├── __init__.py
    ├── page1.py
    ├── page2.py
    ├── page3.py
    └── page4.py
```

*   `app.py`: The main Streamlit application entry point. It handles the sidebar navigation and orchestrates the loading of different pages.
*   `requirements.txt`: Lists all Python dependencies required to run the application.
*   `application_pages/`: A directory containing separate Python modules for each distinct section (page) of the Streamlit application. This promotes code organization and reusability.
    *   `page1.py`: Contains code for data loading, synthetic data generation, and initial exploration.
    *   `page2.py`: Contains code for data pre-processing and stationarity testing.
    *   `page3.py`: Contains code for ARIMAX model estimation, diagnostics, and comparison.
    *   `page4.py`: Contains code for model persistence (saving/downloading the trained model).

## Technology Stack

*   **Streamlit**: For building interactive web applications with Python.
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations, especially in synthetic data generation.
*   **Matplotlib**: For creating static, animated, and interactive visualizations.
*   **StatsModels**: A powerful Python library for statistical modeling, including time series analysis (ARIMAX).
*   **Pickle**: Python's standard module for serializing and de-serializing Python object structures.

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes and ensure they adhere to PEP 8 coding standards.
4.  Write clear, concise commit messages.
5.  Push your branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

### Suggested Improvements:

*   Allow users to upload their own datasets.
*   Implement forecasting capabilities for the trained ARIMAX model.
*   Add more advanced time series diagnostic plots (e.g., QQ plots for residuals).
*   Integrate hyperparameter tuning for ARIMAX models (e.g., using `auto_arima`).
*   Expand the selection of exogenous variables or allow user selection from available columns.
*   Add error handling for invalid user inputs or data issues.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if you have one, otherwise specify "MIT License" or "No specific license for lab use").

## Contact

For any questions or suggestions, please feel free to reach out:

*   **Your Name/Organization Name**
*   **GitHub:** [Your GitHub Profile/Organization Profile](https://github.com/your-username)
*   **Email:** [your.email@example.com](mailto:your.email@example.com)
