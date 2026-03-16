import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Hospital Readmission Predictor", layout="wide")

st.title("Healthcare Analytics Project")
st.subheader("Predicting Hospital Readmissions for Diabetes Patients")

# Load model
with open("outputs/final_xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load original dataset for dropdown values
df = pd.read_csv("data/diabetic_data.csv")
df = df.replace("?", pd.NA)
df = df.drop(columns=["weight", "payer_code", "medical_specialty"])
df = df[df["gender"] != "Unknown/Invalid"]
df = df.drop(columns=["readmitted", "encounter_id", "patient_nbr"])

def get_options(col):
    return sorted(df[col].dropna().astype(str).unique().tolist())

st.markdown("Enter key patient and treatment details to estimate the probability of readmission within 30 days.")

col1, col2 = st.columns(2)

with col1:
    race = st.selectbox("Race", get_options("race"))
    gender = st.selectbox("Gender", get_options("gender"))
    age = st.selectbox("Age Group", get_options("age"))
    time_in_hospital = st.number_input("Time in Hospital", min_value=1, value=3)
    num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, value=40)
    num_medications = st.number_input("Number of Medications", min_value=0, value=10)
    number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, value=5)
    number_inpatient = st.number_input("Previous Inpatient Visits", min_value=0, value=0)

with col2:
    admission_type_id = st.number_input("Admission Type ID", min_value=1, value=1)
    discharge_disposition_id = st.number_input("Discharge Disposition ID", min_value=1, value=1)
    admission_source_id = st.number_input("Admission Source ID", min_value=1, value=1)
    diag_1 = st.text_input("Primary Diagnosis Code", "428")
    diag_2 = st.text_input("Secondary Diagnosis Code", "250")
    diag_3 = st.text_input("Tertiary Diagnosis Code", "401")
    A1Cresult = st.selectbox("A1C Result", get_options("A1Cresult"))
    max_glu_serum = st.selectbox("Max Glucose Serum", get_options("max_glu_serum"))
    insulin = st.selectbox("Insulin", get_options("insulin"))
    change = st.selectbox("Change in Medication", get_options("change"))
    diabetesMed = st.selectbox("Diabetes Medication", get_options("diabetesMed"))

if st.button("Predict Readmission Risk"):
    input_data = pd.DataFrame([{
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": 0,
        "num_medications": num_medications,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": number_inpatient,
        "diag_1": diag_1,
        "diag_2": diag_2,
        "diag_3": diag_3,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        "metformin": "No",
        "repaglinide": "No",
        "nateglinide": "No",
        "chlorpropamide": "No",
        "glimepiride": "No",
        "acetohexamide": "No",
        "glipizide": "No",
        "glyburide": "No",
        "tolbutamide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "acarbose": "No",
        "miglitol": "No",
        "troglitazone": "No",
        "tolazamide": "No",
        "examide": "No",
        "citoglipton": "No",
        "insulin": insulin,
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
        "change": change,
        "diabetesMed": diabetesMed
    }])

    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.write(f"### Readmission Probability: {probability:.2%}")

    if prediction == 1:
        st.error("High risk of readmission within 30 days.")
    else:
        st.success("Lower risk of readmission within 30 days.")