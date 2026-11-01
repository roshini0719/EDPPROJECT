import re
from typing import Dict, List

import nltk

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except Exception:
    stopwords = None
    word_tokenize = None

from ml_model import predict_job_role

COMMON_SKILLS = [
    "Python", "Java", "C", "C++", "SQL", "HTML", "CSS", "JavaScript", "React",
    "Machine Learning", "Deep Learning", "Artificial Intelligence", "NLP", "TensorFlow",
    "PyTorch", "Pandas", "NumPy", "Scikit-learn", "Git", "GitHub", "MySQL",
    "MongoDB", "AWS", "Docker", "Tableau", "Statistics", "Excel", "Flask", "Django",
    "Data Analysis", "Natural Language Processing"
]

ROLE_SKILLS = {
    "Data Scientist": ["Python", "SQL", "Pandas", "NumPy", "Machine Learning", "Statistics", "Data Visualization"],
    "Machine Learning Engineer": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "NumPy", "SQL"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Git", "GitHub", "Responsive Design"],
    "Software Developer": ["Python", "Java", "SQL", "Git", "GitHub", "Problem Solving", "Software Development"],
    "Data Analyst": ["Python", "SQL", "Excel", "Statistics", "Data Visualization", "Pandas"],
}


def ensure_nltk_data():
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
    except Exception:
        return False
    return True


def clean_resume_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s+\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if stopwords is not None and word_tokenize is not None:
        try:
            tokens = word_tokenize(text)
            filtered = [token for token in tokens if token not in stopwords.words("english")]
            return " ".join(filtered)
        except Exception:
            pass

    tokens = text.split()
    return " ".join(tokens)


def normalize_skill(skill: str) -> str:
    return skill.lower().replace("-", " ").replace("+", " ")


def extract_skills(text: str) -> List[str]:
    cleaned = (text or "").lower()
    found = []
    for skill in COMMON_SKILLS:
        skill_key = normalize_skill(skill)
        if skill_key in cleaned or skill_key.replace(" ", "") in cleaned:
            found.append(skill)
    return found


def calculate_resume_score(data: Dict[str, str]) -> int:
    weights = {
        "full_name": 10,
        "career_objective": 10,
        "degree": 15,
        "technical_skills": 15,
        "projects": 15,
        "experience": 10,
        "certifications": 10,
        "achievements": 5,
        "contact_links": 5,
        "keyword_relevance": 5,
    }

    score = 0
    if data.get("full_name", "").strip():
        score += weights["full_name"]
    if data.get("career_objective", "").strip():
        score += weights["career_objective"]
    if data.get("degree", "").strip() or data.get("college", "").strip():
        score += weights["degree"]
    if data.get("technical_skills", "").strip():
        score += weights["technical_skills"]
    if data.get("projects", "").strip():
        score += weights["projects"]
    if data.get("experience", "").strip():
        score += weights["experience"]
    if data.get("certifications", "").strip():
        score += weights["certifications"]
    if data.get("achievements", "").strip():
        score += weights["achievements"]
    if data.get("linkedin", "").strip() or data.get("github", "").strip():
        score += weights["contact_links"]

    resume_text = " ".join(
        [
            data.get("career_objective", ""),
            data.get("technical_skills", ""),
            data.get("projects", ""),
            data.get("experience", ""),
            data.get("certifications", ""),
            data.get("achievements", ""),
        ]
    )
    detected = extract_skills(resume_text)
    if detected:
        score += min(weights["keyword_relevance"], int((len(detected) / max(len(COMMON_SKILLS), 1)) * 5))

    return min(score, 100)


def get_required_skills_for_role(role: str) -> List[str]:
    return ROLE_SKILLS.get(role, ROLE_SKILLS["Data Scientist"])


def generate_suggestions(missing_skills: List[str], data: Dict[str, str], predicted_role: str) -> List[str]:
    suggestions = []

    if missing_skills:
        suggestions.append(f"Add {', '.join(missing_skills[:3])} to improve your suitability for {predicted_role}.")
    if not data.get("projects", "").strip():
        suggestions.append("Add at least two academic or personal projects to strengthen your profile.")
    if not data.get("certifications", "").strip():
        suggestions.append("Add certifications related to your target role.")
    if not (data.get("linkedin", "").strip() or data.get("github", "").strip()):
        suggestions.append("Add your GitHub or LinkedIn profile for better professional visibility.")
    if not data.get("career_objective", "").strip():
        suggestions.append("Write a clear career objective mentioning your technical skills and target role.")
    if not data.get("technical_skills", "").strip():
        suggestions.append("Add more technical skills relevant to the selected role.")
    if not suggestions:
        suggestions.append("Your resume is strong. Focus on measurable achievements and stronger project descriptions.")
    return suggestions[:5]


def improve_career_objective(objective: str, predicted_role: str, detected_skills: List[str]) -> str:
    skills_text = ", ".join(detected_skills[:4]) if detected_skills else "technical and analytical skills"
    role_text = predicted_role.lower()

    if not objective or not objective.strip():
        return (
            f"Motivated student with knowledge of {skills_text}, seeking an opportunity to apply "
            f"my technical abilities in a {role_text} role and contribute to innovative projects."
        )

    improved = (
        f"Motivated {role_text} candidate with knowledge of {skills_text}, seeking an opportunity to "
        f"apply practical skills and contribute to impactful {role_text} projects."
    )
    return improved


def analyze_resume(data: Dict[str, str]) -> Dict[str, object]:
    ensure_nltk_data()
    resume_text = " ".join(
        [
            data.get("full_name", ""),
            data.get("career_objective", ""),
            data.get("degree", ""),
            data.get("college", ""),
            data.get("technical_skills", ""),
            data.get("soft_skills", ""),
            data.get("experience", ""),
            data.get("projects", ""),
            data.get("certifications", ""),
            data.get("achievements", ""),
            data.get("linkedin", ""),
            data.get("github", ""),
        ]
    )

    cleaned_text = clean_resume_text(resume_text)
    detected_skills = extract_skills(cleaned_text)

    prediction = predict_job_role(cleaned_text)
    predicted_role = prediction["job_role"]
    confidence = prediction["confidence"]

    required_skills = get_required_skills_for_role(predicted_role)
    user_skills = set(skill.lower() for skill in detected_skills)
    required_set = {skill.lower() for skill in required_skills}
    missing_skills = [skill for skill in required_skills if skill.lower() not in user_skills]

    score = calculate_resume_score(data)
    improved_objective = improve_career_objective(data.get("career_objective", ""), predicted_role, detected_skills)
    suggestions = generate_suggestions(missing_skills, data, predicted_role)

    return {
        "resume_score": score,
        "predicted_job_role": predicted_role,
        "confidence": confidence,
        "detected_skills": detected_skills,
        "missing_skills": missing_skills,
        "suggestions": suggestions,
        "improved_objective": improved_objective,
        "required_skills": required_skills,
        "resume_text": cleaned_text,
    }
