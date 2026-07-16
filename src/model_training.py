# ==========================================================
# MODEL TRAINING
# ==========================================================

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# ==========================================================
# LOAD CLEANED DATASET
# ==========================================================

df = pd.read_csv("cleaned_dataset.csv")

print("\nDataset Loaded Successfully")
print(df.shape)

# ==========================================================
# ENCODE CATEGORICAL COLUMNS
# ==========================================================

encoder = LabelEncoder()

categorical_columns = [
    "gender",
    "course",
    "internet_access",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df.drop(["exam_score"], axis=1)

y = df["exam_score"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# RANDOM FOREST MODEL
# ==========================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ==========================================================
# PREDICTION
# ==========================================================

prediction = model.predict(X_test)

# ==========================================================
# MODEL EVALUATION
# ==========================================================

mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
rmse = mse ** 0.5
r2 = r2_score(y_test, prediction)

print("\nModel Performance")
print("-" * 40)

print("MAE  :", round(mae,2))
print("MSE  :", round(mse,2))
print("RMSE :", round(rmse,2))
print("R2 Score :", round(r2,2))

# ==========================================================
# SAMPLE PREDICTIONS
# ==========================================================

result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": prediction
})

print("\nSample Predictions")

print(result.head(10))

# ==========================================================
# SAVE MODEL
# ==========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/exam_score_model.pkl")

print("\nModel Saved Successfully")

print("Location : models/exam_score_model.pkl")