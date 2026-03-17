import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.resume_parser import extract_text_from_pdf
from src.data_processing import clean_text
from src.jd_parser import process_job_description
from src.feature_engineering import fit_vectorizer, load_vectorizer, transform_text
from src.similarity import calculate_similarity
from src.utils import extract_skills
from src.recommender import skill_match, final_score, recommend_jobs
from src.model import train_classifier, load_model, predict_role
from src.data_loader import load_resume_data

# ------------------ UI CONFIG ------------------
st.set_page_config(page_title="AI Resume Screening System", layout="wide")

st.title("🧠 AI Resume Screening & Job Recommendation System")

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Options")

train_model_btn = st.sidebar.button("Train Model")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = load_resume_data()
    return df

df = load_data()

# ------------------ TRAIN MODEL ------------------
if train_model_btn:
    st.sidebar.write("Training model...")

    df['clean_resume'] = df['Resume_str'].apply(clean_text)

    vectorizer, X = fit_vectorizer(df['clean_resume'])
    model, acc = train_classifier(X, df['Category'])

    st.sidebar.success(f"Model trained! Accuracy: {acc:.2f}")

# ------------------ LOAD MODEL ------------------
if os.path.exists("models/vectorizer.pkl") and os.path.exists("models/classifier.pkl"):
    vectorizer = load_vectorizer()
    model = load_model()
else:
    vectorizer = None
    model = None

# ------------------ MAIN INPUT ------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload Resume")
    resume_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

with col2:
    st.subheader("📝 Job Description")
    jd_text = st.text_area("Paste Job Description Here")

# ------------------ PROCESS ------------------
if st.button("🚀 Analyze Resume"):

    if not resume_file or not jd_text:
        st.warning("Please upload resume and enter job description.")
    
    else:
        # Extract Resume Text
        resume_text = extract_text_from_pdf(resume_file)

        # Clean Text
        clean_resume = clean_text(resume_text)
        clean_jd = process_job_description(jd_text)

        # Check vectorizer
        if vectorizer is None:
            st.error("⚠️ Please train the model first from sidebar.")
            st.stop()

        # Transform
        resume_vec = transform_text(vectorizer, [clean_resume])
        jd_vec = transform_text(vectorizer, [clean_jd])

        # Similarity Score
        similarity_score = calculate_similarity(resume_vec, jd_vec) * 100

        # Skills
        resume_skills = extract_skills(clean_resume)
        jd_skills = extract_skills(clean_jd)

        skill_percent, missing_skills, matched_skills = skill_match(resume_skills, jd_skills)

        # Final Score
        final = final_score(similarity_score, skill_percent)

        # Predict Role
        if model:
            predicted_role = predict_role(model, vectorizer, clean_resume)
        else:
            predicted_role = "Model not trained"

        # Job Recommendations
        jobs = recommend_jobs(predicted_role)

        # ------------------ OUTPUT ------------------
        st.markdown("## 📊 Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Match Score", f"{similarity_score:.2f}%")
        col2.metric("Skill Match", f"{skill_percent:.2f}%")
        col3.metric("Final Score", f"{final:.2f}%")

        st.markdown("### 🧠 Predicted Role")
        st.success(predicted_role)

        st.markdown("### ✅ Matched Skills")
        st.write(matched_skills if matched_skills else "No matched skills found")

        st.markdown("### ❌ Missing Skills")
        st.write(missing_skills if missing_skills else "No missing skills")

        st.markdown("### 💼 Recommended Jobs")
        st.write(jobs)

        # Decision
        st.markdown("### 🎯 Final Decision")
        if final > 60:
            st.success("Suitable Candidate ✅")
        else:
            st.error("Not Suitable ❌")