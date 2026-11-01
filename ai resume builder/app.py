from flask import Flask, flash, redirect, render_template, request, session, url_for

from resume_analyzer import analyze_resume

app = Flask(__name__)
app.secret_key = "ai-resume-builder-secret-key"


def sample_resume():
    return {
        "full_name": "Roshini",
        "email": "roshini@example.com",
        "phone": "+91 9876543210",
        "location": "Chennai, Tamil Nadu",
        "linkedin": "https://linkedin.com/in/roshini",
        "github": "https://github.com/roshini",
        "career_objective": "I want a job in IT and I love Python, SQL and machine learning.",
        "target_job_role": "Data Scientist",
        "degree": "B.Tech Information Technology",
        "college": "SRM Institute of Science and Technology",
        "graduation_year": "2025",
        "technical_skills": "Python, SQL, Machine Learning, Java, HTML, CSS, Pandas, NumPy",
        "soft_skills": "Communication, Teamwork, Problem Solving",
        "experience": "AIML Internship at TechNova Solutions, worked on data cleaning and predictive modeling.",
        "projects": "Handwritten Digit Recognition using Python and Machine Learning; Sales Forecasting Dashboard using SQL and Excel.",
        "certifications": "AI Fundamentals, Python for Data Science",
        "achievements": "Won AI hackathon finalist, improved model accuracy by 12%.",
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/resume")
def resume_form():
    return render_template("resume_form.html", resume=sample_resume())


@app.route("/analyze", methods=["POST"])
def analyze():
    form_data = request.form.to_dict()

    required_fields = [
        "full_name",
        "email",
        "degree",
        "career_objective",
        "technical_skills",
    ]
    missing = [field for field in required_fields if not str(form_data.get(field, "")).strip()]

    if missing:
        flash(f"Please complete the required fields: {', '.join(missing)}.", "error")
        return render_template("resume_form.html", resume=form_data)

    try:
        result = analyze_resume(form_data)
        session["resume_data"] = form_data
        session["analysis_result"] = result
        return render_template("analysis.html", resume=form_data, result=result)
    except Exception as exc:
        flash(f"Unable to analyze the resume. Please check the form fields and try again. Error: {exc}", "error")
        return render_template("resume_form.html", resume=form_data)


@app.route("/resume-preview")
def resume_preview():
    resume = session.get("resume_data")
    result = session.get("analysis_result")
    if not resume:
        return redirect(url_for("resume_form"))
    return render_template("resume.html", resume=resume, result=result)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
