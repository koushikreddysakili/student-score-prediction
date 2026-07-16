# ==========================================================
# DATA VISUALIZATION
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 50)
print("DATA VISUALIZATION")
print("=" * 50)

# ==========================================================
# CREATE IMAGES FOLDER
# ==========================================================

os.makedirs("images", exist_ok=True)

# ==========================================================
# LOAD CLEANED DATASET
# ==========================================================

df = pd.read_csv("cleaned_dataset.csv")

print("\nDataset Loaded Successfully")
print(df.shape)

# ==========================================================
# HISTOGRAM
# ==========================================================

plt.figure(figsize=(8,5))
plt.hist(df["exam_score"], bins=20)

plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Number of Students")

plt.savefig("images/histogram.png")
plt.show()

# ==========================================================
# BAR CHART
# ==========================================================

course_avg = df.groupby("course")["exam_score"].mean()

plt.figure(figsize=(8,5))
plt.bar(course_avg.index, course_avg.values)

plt.title("Average Exam Score by Course")
plt.xlabel("Course")
plt.ylabel("Average Score")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("images/bar_chart.png")
plt.show()

# ==========================================================
# PIE CHART
# ==========================================================

gender_count = df["gender"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(
    gender_count,
    labels=gender_count.index,
    autopct="%1.1f%%"
)

plt.title("Gender Distribution")

plt.savefig("images/pie_chart.png")
plt.show()

# ==========================================================
# BOXPLOT
# ==========================================================

plt.figure(figsize=(8,5))

sns.boxplot(data=df[["study_hours","sleep_hours","exam_score"]])

plt.title("Study Hours, Sleep Hours and Exam Score")

plt.savefig("images/boxplot.png")
plt.show()

# ==========================================================
# SCATTER PLOT
# ==========================================================

plt.figure(figsize=(8,5))

plt.scatter(
    df["study_hours"],
    df["exam_score"]
)

plt.xlabel("Study Hours")
plt.ylabel("Exam Score")

plt.title("Study Hours vs Exam Score")

plt.savefig("images/scatter_plot.png")
plt.show()

# ==========================================================
# HEATMAP
# ==========================================================

plt.figure(figsize=(8,6))

sns.heatmap(
    df.select_dtypes(include=["int64","float64"]).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("images/heatmap.png")
plt.show()

print("\nAll Charts Created Successfully!")
print("\nCharts saved inside the images folder.")