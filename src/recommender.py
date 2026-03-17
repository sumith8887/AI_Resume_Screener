def skill_match(resume_skills, jd_skills):
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    matched = resume_set.intersection(jd_set)
    missing = jd_set - resume_set

    match_percent = (len(matched) / len(jd_set)) * 100 if jd_set else 0

    return match_percent, list(missing), list(matched)


def final_score(similarity_score, skill_score):
    return (0.7 * similarity_score) + (0.3 * skill_score)


def recommend_jobs(role):
    job_map = {
        "Data Science": ["ML Engineer", "Data Analyst", "AI Engineer"],
        "Java Developer": ["Backend Developer", "Software Engineer"],
        "HR": ["HR Manager", "Recruiter"],
        "Web Designing": ["Frontend Developer", "UI Developer"]
    }

    return job_map.get(role, ["General Role"])