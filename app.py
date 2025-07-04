import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample Data
data = {
    'Study_Hours': [2, 4, 6, 8, 10],
    'Sleep_Hours': [6, 7, 6, 8, 7],
    'Practice_Questions': [5, 10, 15, 20, 30],
    'Attendance': [70, 80, 85, 90, 95],
    'Marks': [45, 55, 65, 80, 95]
}
df = pd.DataFrame(data)

# Model Training
X = df[['Study_Hours', 'Sleep_Hours', 'Practice_Questions', 'Attendance']]
y = df['Marks']
model = LinearRegression()
model.fit(X, y)

# Web App
st.title("🎓 Student Marks Predictor")

study = st.number_input("📘 Study Hours", 0.0, 24.0, step=0.5)
sleep = st.number_input("😴 Sleep Hours", 0.0, 24.0, step=0.5)
practice = st.number_input("📚 Practice Questions", 0, 100, step=1)
attendance = st.number_input("🏫 Attendance (%)", 0, 100, step=1)

if st.button("Predict Marks"):
    input_data = pd.DataFrame([[study, sleep, practice, attendance]], 
                              columns=['Study_Hours', 'Sleep_Hours', 'Practice_Questions', 'Attendance'])
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Marks: {prediction[0]:.2f}")
