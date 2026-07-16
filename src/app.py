import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "exam_score_model.pkl"

model = joblib.load(MODEL_PATH)

# ==========================================
# TITLE
# ==========================================

st.title("🎓 Student Exam Score Prediction")

st.write("Enter the student details below.")

# ==========================================
# USER INPUT
# ==========================================

student_id = st.number_input("Student ID", 1, 50000, 1)

age = st.number_input("Age", 15, 40, 20)

gender = st.selectbox(
    "Gender",
    ["male", "female", "other"]
)

course = st.selectbox(
    "Course",
    ["b.tech", "b.sc", "bca", "diploma"]
)

study_hours = st.slider(
    "Study Hours",
    0.0,
    12.0,
    5.0
)

attendance = st.slider(
    "Class Attendance (%)",
    0.0,
    100.0,
    80.0
)

internet = st.selectbox(
    "Internet Access",
    ["yes", "no"]
)

sleep_hours = st.slider(
    "Sleep Hours",
    1.0,
    12.0,
    7.0
)

sleep_quality = st.selectbox(
    "Sleep Quality",
    ["poor", "average", "good"]
)

study_method = st.selectbox(
    "Study Method",
    ["self study", "group study", "online videos", "coaching"]
)

facility = st.selectbox(
    "Facility Rating",
    ["low", "medium", "high"]
)

difficulty = st.selectbox(
    "Exam Difficulty",
    ["easy", "moderate", "hard"]
)

# ==========================================
# SIMPLE ENCODING
# ==========================================

gender_map = {
    "female":0,
    "male":1,
    "other":2
}

course_map = {
    "b.sc":0,
    "b.tech":1,
    "bca":2,
    "diploma":3
}

internet_map = {
    "no":0,
    "yes":1
}

sleep_map = {
    "average":0,
    "good":1,
    "poor":2
}

study_map = {
    "coaching":0,
    "group study":1,
    "online videos":2,
    "self study":3
}

facility_map = {
    "high":0,
    "low":1,
    "medium":2
}

difficulty_map = {
    "easy":0,
    "hard":1,
    "moderate":2
}

# ==========================================
# PREDICT
# ==========================================

if st.button("Predict Exam Score"):

    data = pd.DataFrame({

        "student_id":[student_id],
        "age":[age],
        "gender":[gender_map[gender]],
        "course":[course_map[course]],
        "study_hours":[study_hours],
        "class_attendance":[attendance],
        "internet_access":[internet_map[internet]],
        "sleep_hours":[sleep_hours],
        "sleep_quality":[sleep_map[sleep_quality]],
        "study_method":[study_map[study_method]],
        "facility_rating":[facility_map[facility]],
        "exam_difficulty":[difficulty_map[difficulty]]

    })

    prediction = model.predict(data)

    st.success(f"Predicted Exam Score : {prediction[0]:.2f}")