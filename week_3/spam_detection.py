
# Week 3: SMS Spam Detection

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Load Dataset

df = pd.read_csv("spam_sms.csv", encoding="latin-1")

# Keep only the required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

print("=" * 50)
print("SMS SPAM DATASET")
print("=" * 50)

print(df.head())

# Dataset Information

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSpam and Ham Count:")
print(df['label'].value_counts())

# ----------------------------------
# Features and Labels
# ----------------------------------

X = df['message']
y = df['label']

# ----------------------------------
# Convert Text into Numbers
# ----------------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

# ----------------------------------
# Train-Test Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Records:", X_train.shape[0])
print("Testing Records :", X_test.shape[0])

# ----------------------------------
# Train Model
# ----------------------------------

model = MultinomialNB()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ----------------------------------
# Prediction
# ----------------------------------

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

# ----------------------------------
# Accuracy
# ----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ----------------------------------
# Test Message 1
# ----------------------------------

sms1 = ["Congratulations! You have won a free iPhone. Click here to claim now."]

sms1 = vectorizer.transform(sms1)

prediction = model.predict(sms1)

print("\nMessage 1 Prediction:", prediction[0])

# ----------------------------------
# Test Message 2
# ----------------------------------

sms2 = ["Hi, are we meeting at 5 PM today?"]

sms2 = vectorizer.transform(sms2)

prediction = model.predict(sms2)

print("Message 2 Prediction:", prediction[0])