from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATASET_PATH = Path(__file__).resolve().parent / "data" / "job_roles.csv"


def ensure_dataset():
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists():
        return

    default_rows = [
        "python pandas numpy machine learning statistics data science", "Data Scientist",
        "python tensorflow deep learning neural networks", "Machine Learning Engineer",
        "html css javascript frontend react web design", "Web Developer",
        "java spring boot sql backend api", "Software Developer",
        "python excel sql statistics dashboard analysis", "Data Analyst",
        "machine learning python nlp pandas scikit learn", "Data Scientist",
        "python sql tableau data visualization analytics", "Data Analyst",
        "python flask html css javascript backend", "Web Developer",
        "tensorflow pytorch deep learning computer vision", "Machine Learning Engineer",
    ]

    rows = [
        {"text": default_rows[i], "job_role": default_rows[i + 1]}
        for i in range(0, len(default_rows), 2)
    ]
    pd.DataFrame(rows).to_csv(DATASET_PATH, index=False)


def train_model():
    ensure_dataset()
    try:
        df = pd.read_csv(DATASET_PATH)
        if df.empty or "text" not in df.columns or "job_role" not in df.columns:
            raise ValueError("Dataset is missing required columns.")

        X = df["text"].fillna("")
        y = df["job_role"].fillna("Data Scientist")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X_train_vec = vectorizer.fit_transform(X_train)

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train_vec, y_train)
        return vectorizer, model
    except Exception:
        return None, None


def predict_job_role(resume_text):
    vectorizer, model = train_model()
    if vectorizer is None or model is None:
        return {"job_role": "Data Scientist", "confidence": 0.45}

    try:
        transformed = vectorizer.transform([resume_text])
        prediction = model.predict(transformed)[0]
        probabilities = model.predict_proba(transformed)[0]
        confidence = float(max(probabilities))
        return {"job_role": prediction, "confidence": round(confidence * 100, 2)}
    except Exception:
        return {"job_role": "Data Scientist", "confidence": 0.45}
