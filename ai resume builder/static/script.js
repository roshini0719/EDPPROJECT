document.addEventListener('DOMContentLoaded', function () {
  const loadSampleButton = document.getElementById('loadSample');
  if (!loadSampleButton) return;

  loadSampleButton.addEventListener('click', function () {
    const sample = {
      full_name: 'Roshini',
      email: 'roshini@example.com',
      phone: '+91 9876543210',
      location: 'Chennai, Tamil Nadu',
      linkedin: 'https://linkedin.com/in/roshini',
      github: 'https://github.com/roshini',
      career_objective: 'I want a job in IT and I love Python, SQL and machine learning.',
      target_job_role: 'Data Scientist',
      degree: 'B.Tech Information Technology',
      college: 'SRM Institute of Science and Technology',
      graduation_year: '2025',
      technical_skills: 'Python, SQL, Machine Learning, Java, HTML, CSS, Pandas, NumPy',
      soft_skills: 'Communication, Teamwork, Problem Solving',
      experience: 'AIML Internship at TechNova Solutions, worked on data cleaning and predictive modeling.',
      projects: 'Handwritten Digit Recognition using Python and Machine Learning; Sales Forecasting Dashboard using SQL and Excel.',
      certifications: 'AI Fundamentals, Python for Data Science',
      achievements: 'Won AI hackathon finalist, improved model accuracy by 12%.'
    };

    Object.entries(sample).forEach(([key, value]) => {
      const field = document.querySelector(`[name="${key}"]`);
      if (field) field.value = value;
    });
  });
});
