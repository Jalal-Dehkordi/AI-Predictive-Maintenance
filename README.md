# AI-Predictive-Maintenance
This repository is for modeling industrial time series data to predict equipment failure.

# Predictive Maintenance for Industrial Machines

A comprehensive machine learning project for predicting failures in industrial machinery before they occur, utilizing the Microsoft Azure Predictive Maintenance dataset.

**Live Demo Preview | [https://www.kaggle.com/code/jalaldehkordi/failsafe-ai]**  

## 🎯 Project Overview

This project tackles a **multi-class classification** problem to predict which component of an industrial machine is likely to fail. By analyzing sensor data (voltage, rotation, pressure, vibration), error logs, and maintenance records, the model aims to transition maintenance strategies from reactive to **predictive**, reducing downtime and operational costs.

The project follows a complete ML pipeline:
-   **Data Integration & Cleaning**
-   **Exploratory Data Analysis (EDA)**
-   **Advanced Feature Engineering & Time-Series Analysis**
-   **Hybrid Feature Selection**
-   **Model Training & Evaluation (4 Different Models)**
-   **Results Analysis & Interpretation**

## 📊 Dataset

The project uses the public **Microsoft Azure Predictive Maintenance** dataset (https://kaggle/input/microsoft-azure-predictive-maintenance)

The dataset consists of 5 CSV files collected from 100 machines in 2015:
-   `PdM_telemetry.csv`: Sensor data (876,100 records)
-   `PdM_errors.csv`: Error logs (3,919 records)
-   `PdM_maint.csv`: Maintenance records (3,286 records)
-   `PdM_failures.csv`: Failure records (761 records) - **Target Labels**
-   `PdM_machines.csv`: Machine metadata (100 records)

## 🛠️ Technical Implementation

### 1. Feature Engineering
The core of this project lies in creating powerful time-series features:
-   **Time-based features:** Extracted `hour`, `dayofweek`, `month`, etc. from datetime.
-   **Rolling Window Features:** Created for sensor data using windows of [3, 6, 12, 24, 72] hours:
    -   Moving Average (`_ma_`)
    -   Standard Deviation (`_std_`)
    -   Minimum & Maximum (`_min_`, `_max_`)
    -   Difference from Mean (`_diff_`)
-   **Event Count Features:** Rolling count of errors, maintenance, and failures.
-   **Time Since Last Event:** e.g., `hours_since_last_error`.

### 2. Feature Selection
A hybrid approach was used to select the most relevant 30 features out of 137+:
1.  **ANOVA (F-test)** for initial filtering (Select 50 best features).
2.  **Random Forest Feature Importance** for final selection (Top 30 features).

### 3. Models & Evaluation
Four machine learning models were trained and compared:

| Model | Key Characteristics | Best Performance |
| :--- | :--- | :--- |
| **Random Forest** | 100 estimators, max depth=10 | High Accuracy & Robustness |
| **XGBoost** | Gradient Boosting, max depth=6 | **Best F1-Score** |
| **LightGBM** | Lightweight Gradient Boosting | Fast Training |
| **MLP** | Neural Network (100, 50 layers) | Scalable to larger data |

-   **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score (Weighted).
-   **Data Split:** 70% Train, 15% Validation, 15% Test.

## 📈 Results

The models achieved high accuracy in predicting the failure of 4 different components (`comp1`, `comp2`, `comp3`, `comp4`) versus normal operation (`no_failure`).

**Key Findings:**
-   The **XGBoost** model slightly outperformed others on the validation set based on F1-Score.
-   Rolling window features significantly improved model performance.
-   The project demonstrates a viable and effective approach to predictive maintenance.

