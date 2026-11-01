# AI Resume Builder using NLP and Machine Learning

This is a simple Flask-based AI resume builder designed for a beginner-friendly AIML academic project. The application helps students create a resume, analyze it using NLP techniques, predict a suitable job role, and suggest improvements.

## Project Overview

This project demonstrates the following concepts:

### Artificial Intelligence
The application provides intelligent recommendations by checking the user's resume information against common skill patterns and job-role requirements. It highlights low-confidence areas, suggests improvements, and improves the career objective using predefined AI-inspired rules.

### Machine Learning
The project uses a small training dataset and a Logistic Regression classifier with TF-IDF vectorization to predict the most suitable job role. The model is trained on a simple sample dataset of skills and job categories such as Data Scientist, Data Analyst, Web Developer, and Machine Learning Engineer.

### Natural Language Processing
The resume text is cleaned and preprocessed using lowercasing, punctuation removal, tokenization, and stop-word filtering. This allows the system to detect skills and analyze the overall resume content.

### TF-IDF
TF-IDF converts text into numerical features so the ML model can learn patterns from the resume keywords and make predictions. It gives more importance to words that are more distinctive in each role category.

### Skill Gap Analysis
The app compares the skills detected in the user's resume with the required skills for the predicted job role. Missing skills are shown and used to generate suggestions such as learning new technologies or adding stronger project details.

## Project Structure

```text
AI-Resume-Builder/
├── app.py
├── ml_model.py
├── resume_analyzer.py
├── requirements.txt
├── README.md
├── data/
│   └── job_roles.csv
├── templates/
│   ├── index.html
│   ├── resume_form.html
│   ├── analysis.html
│   └── resume.html
├── static/
│   ├── style.css
│   └── script.js
└── model/
```

## How to Run

1. Open the project folder in VS Code.
2. Create a Python virtual environment if needed.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the application:

```bash
python app.py
```

5. Open this URL in the browser:

```text
http://127.0.0.1:5000/
```

## Sample Test Data

The application includes a sample resume for testing.

- Name: Roshini
- Degree: B.Tech Information Technology
- Skills: Python, Java, SQL, Machine Learning, HTML, CSS
- Projects: Handwritten Digit Recognition
- Certification: AI Fundamentals
- Experience: AIML Internship

## Features

- Resume form with personal details and academic information
- AI resume analysis
- Resume score from 0 to 100
- Job role prediction using machine learning
- Detected skills and missing skills
- Career objective improvement
- Suggestions for resume enhancement
- Resume preview and print-to-PDF option

## Error Handling

The app includes friendly validation messages for empty fields, missing resume information, invalid input, and ML pipeline issues.

## Notes

This is a simple academic demonstration and should not be treated as a production-grade hiring model.
