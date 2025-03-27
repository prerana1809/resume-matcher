import os
import re

import docx2txt
import PyPDF2
import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Resume Matcher",
    page_icon="icon.png",
    layout="wide"
)

# Load NLP model (to analyze english text)
nlp = spacy.load("en_core_web_sm")

# Skill database and mappings
SKILL_EQUIVALENCE = {'mysql': 'sql', 'js': 'javascript', 'tf': 'tensorflow', 'torch': 'pytorch', 'c++': 'cpp', 'c#': 'csharp'}
CATEGORY_MAP = {
    "Software Engineering": {"python", "java", "c++", "c#", "go", "swift", "kotlin", "ruby", "scala"},
    "Web Development": {"html", "css", "javascript", "react", "angular", "vue", "typescript", "node.js", "django", "flask"},
    "Data Science": {"pandas", "numpy", "matplotlib", "scikit-learn", "tensorflow", "pytorch", "seaborn"},
    "DevOps & Cloud": {"docker", "kubernetes", "AWS", "azure", "gcp", "terraform", "ansible", "cloud", "aws"},
    "Cybersecurity": {"burp suite", "wireshark", "nmap", "metasploit"},
    "Database Management": {"sql", "mysql", "postgresql", "mongodb", "oracle", "redis"},
    "Machine Learning & AI": {"tensorflow", "pytorch", "scikit-learn", "opencv", "huggingface", "bert", "ml", "ai", "r", "python"},
    "Business Intelligence": {"power bi", "tableau", "excel", "looker"}
}

# Function to extract text from file
def extract_text_from_file(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    try:
        if file_extension == ".pdf":
            reader = PyPDF2.PdfReader(file)
            text = " ".join([page.extract_text() or "" for page in reader.pages])
        elif file_extension in [".docx", ".doc"]:
            text = docx2txt.process(file)
        else:
            return "Unsupported file format"
        return text.strip() if text.strip() else "None"
    except Exception as e:
        return f"Error reading file: {e}"

# Preprocessing function
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r'\r\n|\n|\r', ' ', text)  # Remove newlines
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text if text.strip() else "None"

# Extract skills from text
def extract_skills(text):
    text = preprocess_text(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    all_skills = {skill for skills in CATEGORY_MAP.values() for skill in skills}
    extracted_skills = {SKILL_EQUIVALENCE.get(word, word) for word in tokens if word in all_skills}
    return list(extracted_skills) if extracted_skills else ["None"]

# Categorize skills
def categorize_skills(skill_list):
    categorized_skills = {}
    for category, skills in CATEGORY_MAP.items():
        matched = [skill for skill in skill_list if skill in skills]
        if matched:
            categorized_skills[category] = matched
    return categorized_skills if categorized_skills else {"Uncategorized": ["None"]}

# Calculate resume score
def calculate_resume_score(resume_skills, job_skills):
    resume_skills = list(set(resume_skills))
    job_skills = list(set(job_skills))
    
    if job_skills == ["None"]:
        return 0, set()
    
    matched_skills = set(resume_skills) & set(job_skills)
    jaccard_similarity = len(matched_skills) / len(set(job_skills)) if job_skills else 0
    
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([" ".join(resume_skills), " ".join(job_skills)])
        cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    except ValueError:
        cosine_sim = 0
    
    final_score = (5 * jaccard_similarity) + (5 * cosine_sim)
    return round(final_score, 2), matched_skills

# Streamlit Page Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Match Resume"])

# Home Page
if page == "Home":
    st.title("Welcome to Resume Matcher! 🎯")
    st.write("This system helps you match your resume with job descriptions by analyzing skills and relevance")
    st.image("home.jpg", caption="Resume Matcher")

# About Page
elif page == "About":
    st.title("About Resume Matcher")
    st.write("""
    This system works by extracting relevant skills from your resume and comparing them with the skills required in a job description.
    
    **How it Works:**
    1. Upload your resume in PDF or DOCX format.
    2. Enter the job description you want to match against.
    3. The system extracts key skills from both documents.
    4. NLP techniques (Cosine Similarity, Jaccard Similarity and TF-IDF Vectorization) are used to calculate a match score.
    5. The results display your extracted skills, categorized skills (general role), and final resume score.
    """)

# Match Resume Page
elif page == "Match Resume":
    st.title("Match Your Resume with a Job Description")
    
    uploaded_file = st.file_uploader("Upload your Resume (PDF/DOCX)", type=["pdf", "docx"])
    job_description = st.text_area("Enter Job Description")
    
    if st.button("Match Resume"):
        if uploaded_file and job_description:
            resume_text = extract_text_from_file(uploaded_file)
            cleaned_jd = preprocess_text(job_description)
            
            resume_skills = extract_skills(resume_text)
            job_skills = extract_skills(cleaned_jd)
            
            categorized_resume_skills = categorize_skills(resume_skills)
            categorized_job_skills = categorize_skills(job_skills)
            
            resume_score, matched_skills = calculate_resume_score(resume_skills, job_skills)
            
            st.subheader("Results")
            st.write(f"**Extracted Resume Skills:** {resume_skills}")
            st.write(f"**Categorized Resume Skills:** {categorized_resume_skills}")
            st.write(f"**Extracted Job Skills:** {job_skills}")
            st.write(f"**Categorized Job Skills:** {categorized_job_skills}")
            st.write(f"**Matched Skills:** {matched_skills}")
            st.write(f"**Final Resume Score:** {resume_score} / 10")
        else:
            st.warning("Please upload a resume and enter a job description.")
