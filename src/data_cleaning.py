import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("DATA CLEANING")
print("=" * 60)

# =====================================================
# Load Dataset
# =====================================================

current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(current_folder, "Exam_Score_Prediction.csv")

df = pd.read_csv(dataset_path)

print("\nDataset Loaded Successfully")

# =====================================================
# Missing Values
# =====================================================

print("\nMissing Values")
print(df.isnull().sum())

# =====================================================
# Duplicate Rows
# =====================================================

print("\nDuplicate Rows")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

# =====================================================
# Fill Missing Values
# =====================================================

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# =====================================================
# Encode Categorical Columns
# =====================================================

encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

print("\nDataset Shape After Cleaning")
print(df.shape)

print("\nData Types")
print(df.dtypes)

# =====================================================
# Save Cleaned Dataset
# =====================================================

cleaned_path = os.path.join(current_folder, "cleaned_dataset.csv")

df.to_csv(cleaned_path, index=False)

print("\nCleaned Dataset Saved Successfully")
print(cleaned_path)

print("\nData Cleaning Completed Successfully")