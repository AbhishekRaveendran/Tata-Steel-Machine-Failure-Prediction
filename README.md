# 🏭 Tata Steel | Machine Failure Prediction

An AI-powered predictive maintenance system to forecast machine failures in a manufacturing environment using real-world sensor data. This project aims to minimize downtime, reduce costs, and enable proactive maintenance using advanced regression and classification models.

---

## 🔍 Project Overview

- **Type**: Exploratory Data Analysis, Regression & Classification  
- **Contributor**: [Abhishek Raveendran c t](https://github.com/AbhishekRaveendran)  
- **Dataset Size**: 136,429 rows with numerical and categorical features  
- **Tech Stack**: Python, Pandas, Scikit-learn, XGBoost, SHAP, SMOTE, Matplotlib, Seaborn, PCA  

---

## 🎯 Objective

To predict machine failures (binary: 0 = no failure, 1 = failure) by building robust ML models that:
- Detect potential failures in advance
- Reduce unplanned downtimes
- Optimize maintenance schedules

---

## 📂 Dataset Description

- **Numerical Features**:  
  - Air temperature [K]  
  - Process temperature [K]  
  - Rotational speed [rpm]  
  - Torque [Nm]  
  - Tool wear [min]

- **Categorical Feature**:  
  - Type (tool category)

- **Target Variables**:  
  - Machine failure (Binary: 0 or 1)  
  - Failure types (TWF, HDF, PWF, OSF, RNF)

---

## 🧪 Workflow

### 1. 🧭 Exploratory Data Analysis (EDA)
- Imbalance detection, outlier analysis, and feature distribution  
- Correlation mapping (e.g., temperature & torque)  
- Found Rotational Speed to be highly skewed

### 2. 🧹 Preprocessing
- **Feature Engineering**:  
  - `Temp_Difference = ProcessTemp - AirTemp`  
  - `Power = Torque × RPM`  
  - `Wear_Rate = ToolWear / RPM`
- **Selection & Transformation**:  
  - `SelectKBest` for top features  
  - Log transform for skewed `Power` → `Power_Log`  
  - StandardScaler + PCA (95% retained variance)
- **Splitting & Balancing**:  
  - Stratified 80-20 train-test split  
  - Applied **SMOTE** to handle class imbalance

---

## 🤖 Machine Learning Models

| Model              | Optimization       | Notes                          |
|-------------------|--------------------|--------------------------------|
| Logistic Regression | GridSearchCV       | Good baseline                  |
| Random Forest       | RandomizedSearchCV | Balanced accuracy/performance |
| XGBoost             | GridSearchCV       | ✅ **Best performer**          |

---

## 📈 Evaluation Metrics

- **Primary**: Recall, F1 Score, ROC AUC  
- **Why?**: Accuracy is misleading in imbalanced data. The business goal is to **catch failures early** (maximize recall), and maintain trust (F1).

---

## ✅ Final Model: XGBoost

- Tuned with GridSearchCV  
- Highest Recall & F1  
- SHAP Analysis:  
  - Top predictors: `Power_Log`, `Tool wear [min]`

---

## 💼 Business Impact

| Metric            | Value                         |
|------------------|-------------------------------|
| Failure Detection | 80%                           |
| Monthly Downtime Cost Saved | ~$80,000                |
| False Positive Cost | ~$700/month (500/check)     |
| Compared To:     | Logistic Regression: $62,500 saved <br> Random Forest: $69,050 saved |

---

## 🔍 Problem Statement

> In the manufacturing sector, maintaining the efficiency and reliability of machinery is critical. TATA Steel aims to leverage data analytics and machine learning to predict machine failures in advance. This enables proactive maintenance, minimizes downtime, and improves overall production efficiency.

The dataset represents various operational parameters and failure types. By analyzing this data, we can build reliable models to prevent potential failures before they occur.

---

## 📦 Tech Stack

- Python (Pandas, NumPy)
- Scikit-learn
- XGBoost
- SHAP (Model Explainability)
- SMOTE (Imbalance Handling)
- PCA
- Matplotlib, Seaborn (EDA & Visualization)
- Jupyter Notebook / VS Code / PyCharm

---
