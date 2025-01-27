import streamlit as st
import re
#import csv
#import io
import hashlib
import pandas as pd
import s3fs
import boto3
import time
from rapidfuzz import fuzz

# AWS S3 and Textract Configuration
try:
    S3_BUCKET_NAME = st.secrets["default"]["bucket_name"]
    AWS_ACCESS_KEY = st.secrets["default"]["aws_access_key"]
    AWS_SECRET_KEY = st.secrets["default"]["aws_secret_key"]
    AWS_REGION = st.secrets["default"]["region"]  # Adding region for Textract
except KeyError as e:
    st.error(f"Missing secret key: {e}")
    st.stop()


#st.write("Bucket Name:", S3_BUCKET_NAME)
#st.write("Access Key:", AWS_ACCESS_KEY)
#st.write("Secret Key:", AWS_SECRET_KEY)

# Set up s3fs filesystem
fs = s3fs.S3FileSystem(key=AWS_ACCESS_KEY, secret=AWS_SECRET_KEY)

# Initialize AWS Textract client
textract_client = boto3.client('textract', aws_access_key_id=AWS_ACCESS_KEY,
                               aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

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
def load_pdf_with_textract(file):
    """Extracts text from the PDF using AWS Textract."""
    # Upload the file to S3 before calling Textract
    s3_url = upload_to_s3(file, file.name)
    if s3_url:
        # Start document text detection
        response = textract_client.start_document_text_detection(
            DocumentLocation={'S3Object': {'Bucket': S3_BUCKET_NAME, 'Name': file.name}})
        
        # Get the job ID
        job_id = response['JobId']
        
        # Poll for job completion
        st.write("processing with Textract... please wait.")
        while True:
            result = textract_client.get_document_text_detection(JobId=job_id)
            status = result['JobStatus']
            
            if status == 'SUCCEEDED':
                # Extract text from Textract response
                text = ''
                for item in result['Blocks']:
                    if item['BlockType'] == 'LINE':
                        text += item['Text'] + '\n'
                return text
            elif status == 'FAILED':
                st.error("textract job failed.")
                return ''
            else:
                # Wait for a few seconds before checking the job status again
                time.sleep(5)
                st.write("waiting for Textract job to complete...")

    return ''



def extract_skills(text):
    """Extracts skills from the text based on the 'Skills' section."""
    skills_match = re.search(r"SKILLS\s*([\s\S]*?)(?=(WORK EXPERIENCE|PERSONAL PROJECTS|$))", text, re.DOTALL | re.IGNORECASE)
    if skills_match:
        skills_text = skills_match.group(1)
        skills = re.split(r"[\u2022\n,]+", skills_text)
        return clean_skills([skill.strip() for skill in skills if skill.strip()])
    return []

def clean_skills(raw_skills):
    """Clean and normalize extracted skills."""
    
    cleaned = []
    for skill in raw_skills:
        # Remove extra symbols and normalize text
        cleaned_skill = re.sub(r"^\s*-?\s*", "", skill)
        cleaned_skill = re.sub(r"\(.*?\)", "", cleaned_skill).strip()
        words = re.split(r"[,\s:/;]+", cleaned_skill)
        cleaned.extend(word for word in words if word)
    # Remove duplicates
    return list(dict.fromkeys(cleaned))

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
    """
    Matches candidate's skills with required and good-to-have skills.
    Uses fuzzy matching 
    """
    # Normalize input data
    candidate_set = set(map(str.lower, candidate_skills))
    required_set = set(map(str.lower, required_skills.split(", ")))
    good_to_have_set = set(map(str.lower, good_to_have_skills.split(", ")) if good_to_have_skills else [])

    # Define matching thresholds
    required_threshold = 85  # Threshold for required skills
    good_to_have_threshold = 85  # Threshold for good-to-have skills

    # Match required skills
    matched_required_skills = set()
    for required in required_set:
        for candidate in candidate_set:
            if fuzz.ratio(required, candidate) >= required_threshold:
                matched_required_skills.add(candidate)

    # Match good-to-have skills
    matched_good_to_have_skills = set()
    for good_to_have in good_to_have_set:
        for candidate in candidate_set:
            if fuzz.ratio(good_to_have, candidate) >= good_to_have_threshold:
                matched_good_to_have_skills.add(candidate)

    # Calculate match percentages
    required_match_percentage = (len(matched_required_skills) / len(required_set)) * 70 if required_set else 0
    good_to_have_match_percentage = (len(matched_good_to_have_skills) / len(good_to_have_set)) * 30 if good_to_have_set else 0
    total_match_percentage = required_match_percentage + good_to_have_match_percentage

    return matched_required_skills, matched_good_to_have_skills, total_match_percentage

# Updating the process_single_pdf function to use the new evaluate_skills
def process_single_pdf(file, required_skills, good_to_have_skills, processed_hashes):
    """Processes a single PDF resume file."""
    file_hash = generate_file_hash(file)
    if file_hash in processed_hashes:
        return None
    
    # Extract text using AWS Textract
    text = load_pdf_with_textract(file)
    data = extract_info(text)



    # Evaluate candidate skills
    matched_required_skills, matched_good_to_have_skills, total_match_percentage = evaluate_skills(
        data["skills"], required_skills, good_to_have_skills
    )

    # Prepare candidate data
    candidate_data = data.copy()
    candidate_data["extracted_text"] = text

    candidate_data["matched_skills"] = ", ".join(matched_required_skills)
    candidate_data["matched_good_to_have_skills"] = ", ".join(matched_good_to_have_skills)
    candidate_data["match_percentage"] = total_match_percentage
    processed_hashes.add(file_hash)

    return candidate_data


def generate_file_hash(file):
    """unique hash for the uploaded file."""
    file_content = file.read()
    file.seek(0)
    return hashlib.md5(file_content).hexdigest()

# Streamlit UI
def main():
    st.title("Carnera Resume Screening with AWS S3 & Textract Integration")
    st.subheader("Upload resumes (PDF format)")

    predefined_skills = ["Python", "SQL", "React", "Django", "Java", "JavaScript", "HTML", "CSS", "Node.js", "AWS", "Git", "Machine Learning", "Data Analysis"]
    selected_required_skills = st.multiselect("Select required skills:", predefined_skills)
    required_skills = ", ".join(selected_required_skills) if selected_required_skills else st.text_area("Enter required skills (comma-separated)")
    good_to_have_skills = st.text_area("Enter good-to-have skills (comma-separated)")

    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf","png"], accept_multiple_files=True)
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

                # display extracted text
                #with st.expander(f"Extracted Text from {uploaded_file.name}"):
                #    st.text_area("Extracted Text", candidate_data.get("extracted_text", ""), height=300)

                # display extracted skills
                #st.write(f"**Extracted Skills from {uploaded_file.name}:**")
                #st.write(", ".join(candidate_data.get("skills", [])))


        ranked_candidates = sorted(all_candidates, key=lambda x: x["match_percentage"], reverse=True)

        if ranked_candidates:
            st.write("**Ranked Candidates:**")
            # Convert the ranked candidates into a DataFrame-better handling
            df = pd.DataFrame([{
                "Filename": candidate["filename"],
                "Email": candidate["email"],
                "Phone": candidate["phone"],
                "Matched Skills": candidate["matched_skills"],
                "Good-to-Have Skills": candidate["matched_good_to_have_skills"],
                "Match Percentage": f"{candidate['match_percentage']:.2f}%",
                #"S3 URL": candidate["s3_url"],
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
