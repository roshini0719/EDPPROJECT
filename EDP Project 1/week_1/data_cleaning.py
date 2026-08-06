import pandas as pd

# Load dataset
df = pd.read_csv("titanic.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# Cleaning
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop("Cabin", axis=1)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# Features and Label
X = df.drop("Survived", axis=1)
y = df["Survived"]

print("\nFeatures Shape:", X.shape)
print("Label Shape:", y.shape)

print("\nStatistics")
print(df.describe())

print("\nColumn Names")
print(df.columns)

print("\nData Types")
print(df.dtypes)