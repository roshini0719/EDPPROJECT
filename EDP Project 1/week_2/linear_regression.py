# Week 2: Linear Regression
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset

df = pd.read_csv("Student_DataSet.csv")

print("=" * 50)
print("STUDENT PERFORMANCE DATASET")
print("=" * 50)

print(df.head())
# Dataset Information


print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())
# Features and Target

X = df[['reading_percentage']]
y = df['math_percentage']

print("\nFeature (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())
# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records :", len(X_test))


# Train Model

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# Prediction


y_pred = model.predict(X_test)

print("\nActual Math Percentages:")
print(y_test.values)

print("\nPredicted Math Percentages:")
print(y_pred)

# Evaluation Metrics


print("\n========== MODEL EVALUATION ==========")

print("Mean Absolute Error :", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error  :", mean_squared_error(y_test, y_pred))
print("R2 Score            :", r2_score(y_test, y_pred))

# Predict New Student


new_student = pd.DataFrame({
    "reading_percentage": [0.85]
})

prediction = model.predict(new_student)

print("\nPrediction")

print("Reading Percentage : 85%")
print("Predicted Math Percentage :", round(prediction[0] * 100, 2), "%")

# Graph


plt.figure(figsize=(8,5))

plt.scatter(
    df["reading_percentage"],
    df["math_percentage"],
    label="Actual Data"
)

plt.plot(
    df["reading_percentage"],
    model.predict(X),
    color="red",
    linewidth=2,
    label="Regression Line"
)

plt.title("Reading Percentage vs Math Percentage")

plt.xlabel("Reading Percentage")

plt.ylabel("Math Percentage")

plt.legend()

plt.grid(True)

plt.show()