import pandas as pd
import os


print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(current_folder, "Exam_Score_Prediction.csv")

df = pd.read_csv(dataset_path)

print("\nDataset Loaded Successfully")

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nFirst Five Records")
print(df.head())

print("\nDataset Information")
print(df.info())