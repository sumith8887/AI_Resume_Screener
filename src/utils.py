# You can expand this list anytime
SKILLS = [
    "python", "java", "sql", "machine learning", "deep learning",
    "nlp", "data analysis", "html", "css", "javascript",
    "react", "node", "flask", "django", "aws", "docker"
]

def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in SKILLS:
        if skill in text:
            found_skills.append(skill)

    return list(set(found_skills))