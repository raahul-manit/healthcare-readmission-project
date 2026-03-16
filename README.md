# 🏥 Hospital Readmission Prediction Dashboard

A dynamic and interactive **machine learning healthcare analytics project** designed to predict whether diabetes patients are likely to be readmitted to the hospital within 30 days.  
This project builds a complete **data pipeline using Python** for data cleaning, preprocessing, model training, and prediction deployment through a **Streamlit web app**.

---


## Short Description / Purpose

The **Hospital Readmission Prediction Dashboard** uses historical diabetes patient hospital records to identify patients at high risk of readmission within 30 days.

It helps analyze:

- Patient admission patterns
- Readmission risk factors
- Diagnosis and treatment-related information
- Medication and hospitalization trends

This project demonstrates a complete **data pipeline + machine learning + deployment workflow** commonly used in real-world healthcare analytics projects.

---

## Dashboard Preview

*(Add your Streamlit app screenshot here after uploading your repo)*

Example:

![Healthcare Dashboard](dashboard_preview.png)

---

## Tech Stack

The project was built using the following tools and technologies:

- 🐍 **Python** – Core programming language for the full pipeline  
- 📊 **Pandas** – Data cleaning, preprocessing, and transformation  
- 🔢 **NumPy** – Numerical operations  
- 🤖 **Scikit-learn** – Preprocessing pipelines, model building, and evaluation  
- 🚀 **XGBoost** – Advanced machine learning model for readmission prediction  
- 🌐 **Streamlit** – Deployment of the prediction app  
- 📓 **Jupyter / Notebook-style workflow** – Analysis and experimentation  
- 📁 **CSV Files** – Dataset storage and processed outputs  
- 🧠 **Machine Learning Pipelines** – Structured preprocessing and training workflow  

---

## Data Source

- **Source:** Diabetes 130-US Hospitals dataset (Kaggle / UCI Repository)  
- **Description:** Historical hospital encounter records for diabetes patients including:

  - Demographic information
  - Admission and discharge details
  - Number of lab procedures
  - Number of medications
  - Diagnosis codes
  - Glucose and A1C test results
  - Medication changes
  - Readmission status

The dataset provides hospital-level patient encounter data used to build predictive healthcare insights.

---

## Features / Highlights

### Business Problem

Hospital readmissions are expensive and often signal gaps in patient care, treatment planning, or discharge management. Predicting which patients are at higher risk of readmission can help hospitals improve care continuity and reduce unnecessary costs.

Questions that analysts might ask include:

- Which patients are most likely to be readmitted within 30 days?
- How do diagnosis history and medication changes affect readmission risk?
- Does inpatient history contribute to higher readmission probability?
- Which machine learning model performs best for this prediction problem?

This project helps answer these questions through machine learning and predictive analytics.

---

### Goal of the Project

The objective of this project is to:

- Build a **data pipeline that processes hospital patient data**
- Clean and prepare the dataset for analysis
- Create a **machine learning classification model**
- Compare multiple models for performance
- Deploy the best model as an **interactive Streamlit app**

---

## Walkthrough of Key Components

### Data Cleaning & Preprocessing
The dataset contains missing values, invalid entries, and noisy columns. These were handled through:

- Replacing `?` values with missing values
- Dropping highly sparse or low-utility columns
- Removing invalid patient records
- Handling missing categorical values
- Encoding categorical variables
- Scaling numerical features

---

### Target Variable Engineering
The original readmission target had three classes:

- `NO`
- `>30`
- `<30`

This was converted into a **binary classification target**:

- `1` → Readmitted within 30 days (`<30`)
- `0` → Not readmitted within 30 days (`NO` and `>30`)

---

### Model Comparison
Three machine learning models were trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost

These models were compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

### Best Model
Among all tested models, **XGBoost** performed best and was selected as the final model for deployment.

---

### Streamlit Prediction App
A **Streamlit web app** was built to allow users to input patient details and receive:

- Predicted readmission probability
- Risk classification output
- Simple healthcare decision-support style result

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.6419 | 0.1657 | 0.5473 | 0.2543 | 0.6411 |
| Random Forest | 0.6259 | 0.1673 | 0.5918 | 0.2609 | 0.6561 |
| XGBoost | 0.6703 | 0.1844 | 0.5711 | 0.2788 | 0.6804 |

---

## Data Pipeline Architecture

The project uses a simple **Python-based machine learning pipeline**:

```text
Raw Hospital Dataset
        ↓
Data Loading
        ↓
Data Cleaning (Pandas)
        ↓
Missing Value Handling
        ↓
Feature Engineering
        ↓
Preprocessing Pipeline
        ↓
Model Training & Comparison
        ↓
Best Model Selection (XGBoost)
        ↓
Model Saving
        ↓
Streamlit Deployment


healthcare-readmission-project/
│
├── data
│   └── diabetic_data.csv
│
├── notebooks
│   ├── test_load.py
│   ├── readmission_analysis.py
│   └── model_comparison.py
│
├── src
│   └── train_xgboost.py
│
├── app
│   └── app.py
│
├── outputs
│   ├── model_comparison_results.csv
│   ├── final_xgboost_metrics.csv
│   └── final_xgboost_model.pkl
│
├── README.md
├── requirements.txt
└── .gitignore
