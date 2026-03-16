# Healthcare Analytics Project: Predicting Hospital Readmissions for Diabetes Patients

## Project Overview
This project is an end-to-end healthcare analytics solution focused on predicting whether a diabetes patient is likely to be readmitted to the hospital within 30 days. The objective is to support early risk identification and improve hospital decision-making using machine learning.

## Problem Statement
Hospital readmissions are costly and often indicate gaps in care continuity. In this project, we use historical hospital encounter data for diabetes patients to predict short-term readmission risk.

## Dataset
- Source: Kaggle / UCI Diabetes 130-US Hospitals dataset
- Records: 100k+ patient encounters
- Features: demographics, admission details, diagnosis codes, lab results, medication changes, inpatient/outpatient history

## Project Workflow
1. Data loading and exploration
2. Data cleaning and preprocessing
3. Handling missing values and invalid entries
4. Converting target variable into binary classification
5. Building baseline and advanced ML models
6. Comparing Logistic Regression, Random Forest, and XGBoost
7. Deploying the best model using Streamlit

## Models Used
- Logistic Regression
- Random Forest
- XGBoost

## Final Model Performance
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.6419 | 0.1657 | 0.5473 | 0.2543 | 0.6411 |
| Random Forest | 0.6259 | 0.1673 | 0.5918 | 0.2609 | 0.6561 |
| XGBoost | 0.6703 | 0.1844 | 0.5711 | 0.2788 | 0.6804 |

## Key Insights
- XGBoost performed best among all tested models.
- The dataset was highly imbalanced, with far fewer readmission cases within 30 days.
- Predicting readmission remains challenging, but the model provides a useful risk estimation baseline.

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib / Seaborn
- VS Code

## Project Structure
```text
healthcare-readmission-project/
│
├── data/
├── notebooks/
├── src/
├── app/
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore