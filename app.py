import streamlit as st
import pickle
import numpy as np

# Load the model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("Health Metric Classifier")
st.write("Enter the required details below to get a prediction.")

# Create input fields based on the model's feature names
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("Age", min_value=0, step=1)

if st.button("Predict"):
    # Arrange features in the order the model expects
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, dpf, age]])
    
    # Make prediction
    prediction = model.predict(features)
    
    st.subheader("Result:")
    if prediction[0] == 1:
        st.error("The model classifies this as Positive.")
    else:
        st.success("The model classifies this as Negative.")
