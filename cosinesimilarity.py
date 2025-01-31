import streamlit as st
import re
import hashlib
import pandas as pd
import s3fs
import boto3
import time
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# AWS S3 and Textract Configuration
try:
    S3_BUCKET_NAME = st.secrets["default"]["bucket_name"]
    AWS_ACCESS_KEY = st.secrets["default"]["aws_access_key"]
    AWS_SECRET_KEY = st.secrets["default"]["aws_secret_key"]
    AWS_REGION = st.secrets["default"]["region"]  # Adding region for Textract
except KeyError as e:
    st.error(f"Missing secret key: {e}")
    st.stop()

# Set up s3fs filesystem
fs = s3fs.S3FileSystem(key=AWS_ACCESS_KEY, secret=AWS_SECRET_KEY)

# Initialize AWS Textract client
textract_client = boto3.client('textract', aws_access_key_id=AWS_ACCESS_KEY,
                               aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)


# Preprocessing Function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Functions for AWS S3 (Using s3fs)
def list_folders_in_s3(bucket_name):
    """List all folders in the S3 bucket."""
    objects = fs.ls(bucket_name)
    folders = {obj.split('/')[1] for obj in objects if '/' in obj}  # Extract unique folder names
    return sorted(folders)

def list_files_in_s3(bucket_name, folder_name):
    """List all files in a given S3 folder."""
    folder_path = f"{bucket_name}/{folder_name}/"
    return [obj.split('/')[-1] for obj in fs.ls(folder_path) if obj.endswith(('pdf', 'png', 'jpg', 'jpeg', 'docx'))]

# Resume processing functions
def load_pdf_with_textract(file_key):
    """Extracts text from the PDF using AWS Textract."""
    response = textract_client.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': S3_BUCKET_NAME, 'Name': file_key}})

    # Get the job ID
    job_id = response['JobId']

    # Poll for job completion
    st.write("Processing with Textract... please wait.")
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
            st.error("Textract job failed.")
            return ''
        else:
            # Wait for a few seconds before checking the job status again
            time.sleep(5)
            st.write("Waiting for Textract job to complete...")
    return ''


# Compute TF-IDF & Cosine Similarity
def compute_similarity(job_description, resumes):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([job_description] + resumes)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarity_scores

# Rank Resumes
def rank_resumes(job_description, resumes):
    preprocessed_resumes = [preprocess_text(resume) for resume in resumes]
    job_desc = preprocess_text(job_description)
    similarity_scores = compute_similarity(job_desc, preprocessed_resumes)
    return sorted(zip(resumes, similarity_scores), key=lambda x: x[1], reverse=True)

# Streamlit UI
def main():
    st.title("Resume Screening with AWS S3 & Textract Integration")
    st.subheader("Process resumes directly from S3 bucket and rank by job description similarity")

    # Job Description input
    job_desc = st.text_area("Enter the Job Description",)

    # List all available folders in S3
    available_folders = list_folders_in_s3(S3_BUCKET_NAME)

    if not available_folders:
        st.error("No folders found in S3 bucket.")
        return

    # Allow users to select multiple folders
    selected_folders = st.multiselect("Select folders to process:", available_folders)

    if selected_folders:
        all_files = []
        for folder in selected_folders:
            files_in_folder = list_files_in_s3(S3_BUCKET_NAME, folder)
            all_files.extend([(folder, f) for f in files_in_folder])

        if not all_files:
            st.write("No resumes found in the selected folder(s).")
            return

        st.write("Resumes found:")
        for folder, file in all_files:
            st.write(f"📁 {folder} - {file}")

        if st.button("Process all selected folders"):
            all_resumes = []
            file_names = []  # To store the filenames for the final result
            for folder, file_name in all_files:
                file_key = f"{folder}/{file_name}"
                st.write(f"Processing: {file_key}")

                text = load_pdf_with_textract(file_key)
                if not text:
                    continue

                all_resumes.append(text)
                file_names.append(file_name) 

            # Rank resumes based on similarity to job description
            ranked_resumes = rank_resumes(job_desc, all_resumes)

            # Prepare results for download
            ranked_data = []
            for idx, (resume, score) in enumerate(ranked_resumes, start=1):
                ranked_data.append({
                    #"Rank": idx,
                    "Filename": file_names[idx - 1],
                    #"Resume Text": resume[:300],  # Show only a preview of the resume
                    "Similarity Score": score,
                })

            df = pd.DataFrame(ranked_data)

            st.table(df)

            csv = df.to_csv(index=False)
            st.download_button("Download Results as CSV", csv, "ranked_resumes.csv", "text/csv")

if __name__ == "__main__":
    main()