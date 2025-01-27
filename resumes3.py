import streamlit as st
import pdfplumber
import re
import csv
import io
import hashlib
import pandas as pd
import s3fs



# AWS S3 Configuration
try:
    S3_BUCKET_NAME = st.secrets["default"]["bucket_name"]
    AWS_ACCESS_KEY = st.secrets["default"]["aws_access_key"]
    AWS_SECRET_KEY = st.secrets["default"]["aws_secret_key"]
except KeyError as e:
    st.error(f"Missing secret key: {e}")
    st.stop() 

#st.write("Bucket Name:", S3_BUCKET_NAME)
#st.write("Access Key:", AWS_ACCESS_KEY)
#st.write("Secret Key:", AWS_SECRET_KEY)



# Set up s3fs filesystem
fs = s3fs.S3FileSystem(key=AWS_ACCESS_KEY, secret=AWS_SECRET_KEY)

# Functions for AWS S3 (Using s3fs)
def upload_to_s3(file, filename):
    """Uploads a file to an S3 bucket using s3fs."""
    if not fs:
        return None 
    try:
        s3_path = f"{S3_BUCKET_NAME}/{filename}"
        with fs.open(s3_path, 'wb') as f:
            f.write(file.getbuffer())  # Writing the file to the S3 bucket
        return f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{filename}"
    except Exception as e:
        st.error(f"Error uploading to S3: {e}")
        return None

# Resume processing functions
def load_pdf(file):
    """Extracts text from the PDF."""
    with io.BytesIO(file.read()) as byte_file:
        with pdfplumber.open(byte_file) as pdf:
            text = ''.join(page.extract_text() for page in pdf.pages)
    return text

def extract_skills(text):
    """Extracts skills from the text based on the 'Skills' section."""
    skills_match = re.search(r"SKILLS\s*(.*?)(?=(COURSEWORK|EXPERIENCE|EDUCATION|PROJECTS|$))", text, re.DOTALL | re.IGNORECASE)
    if skills_match:
        skills_text = skills_match.group(1)
        skills = re.split(r"[\u2022\n,]+", skills_text)
        return clean_skills([skill.strip() for skill in skills if skill.strip()])
    return []

def clean_skills(raw_skills):
    """Clean and normalize extracted skills."""
    cleaned = []
    for skill in raw_skills:
        cleaned_skill = re.sub(r"^\s*-?\s*", "", skill)
        cleaned_skill = re.sub(r"\(.*?\)", "", cleaned_skill).strip()
        words = re.split(r"[,\s:/;]+", cleaned_skill)
        cleaned.extend(word.strip() for word in words if word.strip())
    return list(dict.fromkeys(word.lower() for word in cleaned))

def extract_info(text):
    """Extracts key information (email, phone, etc.)."""
    data = {
        "email": re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text),
        "phone": re.search(r"(\+?\d{1,4}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}", text),
        "skills": extract_skills(text),
    }
    result_data = {}
    for key, value in data.items():
        if key != "skills":
            result_data[key] = value.group(0) if value else "Not available"
        else:
            result_data[key] = value
    return result_data

def evaluate_skills(candidate_skills, required_skills, good_to_have_skills):
    """Matches candidate's skills with required and good-to-have skills."""
    candidate_set = set(map(str.lower, candidate_skills))
    required_set = set(map(str.lower, required_skills.split(", ")))
    good_to_have_set = set(map(str.lower, good_to_have_skills.split(", ")) if good_to_have_skills else [])

    matched_required_skills = candidate_set.intersection(required_set)
    matched_good_to_have_skills = candidate_set.intersection(good_to_have_set)

    match_percentage = (len(matched_required_skills) / len(required_set)) * 70 if required_set else 0
    bonus_percentage = (len(matched_good_to_have_skills) / len(good_to_have_set)) * 30 if good_to_have_set else 0
    total_match_percentage = match_percentage + bonus_percentage

    return matched_required_skills, matched_good_to_have_skills, total_match_percentage

def process_single_pdf(file, required_skills, good_to_have_skills, processed_hashes):
    """Processes a single PDF resume file."""
    file_hash = generate_file_hash(file)
    if file_hash in processed_hashes:
        return None
    
    text = load_pdf(file)
    data = extract_info(text)
    matched_required_skills, matched_good_to_have_skills, total_match_percentage = evaluate_skills(data["skills"], required_skills, good_to_have_skills)

    candidate_data = data.copy()
    candidate_data["matched_skills"] = ", ".join(matched_required_skills)
    candidate_data["matched_good_to_have_skills"] = ", ".join(matched_good_to_have_skills)
    candidate_data["match_percentage"] = total_match_percentage
    processed_hashes.add(file_hash)

    return candidate_data

def generate_file_hash(file):
    """Generates a unique hash for the uploaded file."""
    file_content = file.read()
    file.seek(0)
    return hashlib.md5(file_content).hexdigest()

# Streamlit UI
def main():
    st.title("Carnera Resume Screening with AWS S3 Integration")
    st.subheader("Upload resumes (PDF format)")

    predefined_skills = ["Python", "SQL", "React", "Django", "Java", "JavaScript", "HTML", "CSS", "Node.js", "AWS", "Git", "Machine Learning", "Data Analysis"]
    selected_required_skills = st.multiselect("Select required skills:", predefined_skills)
    required_skills = ", ".join(selected_required_skills) if selected_required_skills else st.text_area("Enter required skills (comma-separated)")
    good_to_have_skills = st.text_area("Enter good-to-have skills (comma-separated)")

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    processed_hashes = set()

    if uploaded_files:
        st.write(f"Processing {len(uploaded_files)} resumes...")
        all_candidates = []
        for uploaded_file in uploaded_files:
            s3_url = upload_to_s3(uploaded_file, uploaded_file.name)
            candidate_data = process_single_pdf(uploaded_file, required_skills, good_to_have_skills, processed_hashes)
            if candidate_data:
                candidate_data["filename"] = uploaded_file.name
                candidate_data["s3_url"] = s3_url
                all_candidates.append(candidate_data)

        ranked_candidates = sorted(all_candidates, key=lambda x: x["match_percentage"], reverse=True)

        if ranked_candidates:
            st.write("**Ranked Candidates:**")
             # Convert the ranked candidates into a DataFrame for better handling
            df = pd.DataFrame([{
                "Filename": candidate["filename"],
                "Email": candidate["email"],
                "Phone": candidate["phone"],
                "Matched Skills": candidate["matched_skills"],
                "Good-to-Have Skills": candidate["matched_good_to_have_skills"],
                "Match Percentage": f"{candidate['match_percentage']:.2f}%",
                "S3 URL": candidate["s3_url"],
            } for candidate in ranked_candidates])
            
            st.table(df)

            # Add a download button for the CSV file
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Ranked Candidates as CSV",
                data=csv,
                file_name="ranked_candidates.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()


