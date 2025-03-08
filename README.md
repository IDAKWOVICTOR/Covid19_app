# COVID-19 Case Prediction and Modelling

## Overview
This repository contains a machine learning pipeline for predicting COVID-19 cases based on clinical symptoms. The project involves data preprocessing, feature engineering, model selection, evaluation, and deployment.

## Table of Contents
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Selection & Training](#model-selection--training)
- [Model Evaluation](#model-evaluation)
- [Model Deployment](#model-deployment)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run this project, install the required dependencies:
```bash
pip install lazypredict shap scikit_learn pandas geopandas matplotlib openpyxl statsmodels
```

## Data Collection
Data was collected using **KoboToolbox** and clinical reports. The dataset contains patient symptoms and test results.

## Data Preprocessing
1. **Handling Missing Values:**
   - Dropped completely empty columns.
   - Replaced 'NO' with `0` and 'YES' with `1`.
   - Converted 'POSITIVE' to `1` and 'NEGATIVE' to `0` in the target column.
   - Replaced 'UNKNOWN' values with `0`.
2. **Feature Selection:**
   - Selected key symptoms based on literature.
   - Checked for multicollinearity using **Variance Inflation Factor (VIF)**.

## Feature Engineering
- Conducted **VIF analysis** to detect multicollinearity.
- Evaluated **feature importance** using:
  - **Random Forest** feature importance scores.
  - **Logistic Regression** coefficients.

## Model Selection & Training
Used **LazyPredict** for quick benchmarking and selected the following models:
- **Logistic Regression**
- **Naïve Bayes**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **Gradient Boosting Classifier**

Each model was trained on an **80-20 split** using stratified sampling.

## Model Evaluation
Performance was measured using:
- **Accuracy, Precision, Recall, F1 Score**
- **ROC-AUC Curve**
- **Cohen Kappa Score & Matthews Correlation Coefficient**
- **Log Loss & Brier Score**

## Model Deployment
The best model can be deployed using **Flask** or **Streamlit**:
- Flask app: `covid19_flask_app.py`
- Streamlit app: `covid19_streamlit_app.py`

## Results
- Identified the best-performing model based on **F1 Score**.
- Logistic Regression **ANOVA-like** analysis was conducted.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Submit a pull request.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

