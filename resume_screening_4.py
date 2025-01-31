import streamlit as st
import re
import boto3
import time
import s3fs
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# AWS S3 and Textract Configuration
try:
    S3_BUCKET_NAME = st.secrets["default"]["bucket_name"]
    AWS_ACCESS_KEY = st.secrets["default"]["aws_access_key"]
    AWS_SECRET_KEY = st.secrets["default"]["aws_secret_key"]
    AWS_REGION = st.secrets["default"]["region"]
except KeyError as e:
    st.error(f"Missing secret key: {e}")
    st.stop()

# Set up S3 file system
fs = s3fs.S3FileSystem(key=AWS_ACCESS_KEY, secret=AWS_SECRET_KEY)

# Initialize AWS clients
textract_client = boto3.client('textract', aws_access_key_id=AWS_ACCESS_KEY,
                               aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)
comprehend_client = boto3.client('comprehend', aws_access_key_id=AWS_ACCESS_KEY,
                                 aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

def list_folders_in_s3(bucket_name):
    """List all folders in the S3 bucket."""
    objects = fs.ls(bucket_name)
    folders = {obj.split('/')[1] for obj in objects if '/' in obj}
    return sorted(folders)

def list_files_in_s3(bucket_name, folder_name):
    """List all files in a given S3 folder."""
    folder_path = f"{bucket_name}/{folder_name}/"
    return [obj.split('/')[-1] for obj in fs.ls(folder_path) if obj.endswith(('pdf', 'png', 'jpg', 'jpeg', 'docx'))]

def extract_text_with_textract(file_key):
    """Extract text from a document using AWS Textract."""
    response = textract_client.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': S3_BUCKET_NAME, 'Name': file_key}})
    job_id = response['JobId']
    
    st.write("Processing with Textract... please wait.")
    while True:
        result = textract_client.get_document_text_detection(JobId=job_id)
        status = result['JobStatus']
        
        if status == 'SUCCEEDED':
            text = '\n'.join([item['Text'] for item in result['Blocks'] if item['BlockType'] == 'LINE'])
            return text.lower()
        elif status == 'FAILED':
            st.error("Textract job failed.")
            return ''
        else:
            time.sleep(5)
            st.write("Waiting for Textract job to complete...")



#def analyze_text_with_comprehend(text):
#    """Analyze text using AWS Comprehend."""
#    response = comprehend_client.detect_entities(Text=text, LanguageCode='en')
#    entities = {entity['Type']: entity['Text'] for entity in response['Entities']}
#    return entities

def extract_contact_details(text):
    """Extract phone numbers and emails from text."""
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    phone_numbers = re.findall(r'\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,9}', text)
    return emails, phone_numbers

def compute_cosine_similarity(job_description, resume_texts):
    """Compute cosine similarity between job description and resume texts."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([job_description] + resume_texts)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return cosine_sim[0]

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w.lower()) for w in words if w.lower() not in stop_words]
    return " ".join(cleaned_words)


def main():
    st.title("Resume Processing with AWS Textract & Comprehend")
    st.subheader("Extract text and analyze entities from resumes in S3")
    
    job_description = st.text_area("Enter Job Description:", "").lower()
    st.write(f"Processed Job Description: {job_description}")
    # Preprocess the job description
    job_description = preprocess_text(job_description)
    
    available_folders = list_folders_in_s3(S3_BUCKET_NAME)
    if not available_folders:
        st.error("No folders found in S3 bucket.")
        return
    
    st.write(f"Available Folders: {available_folders}")

    selected_folders = st.multiselect("Select folders to process:", available_folders)
    
    if selected_folders:
        all_files = [(folder, file) for folder in selected_folders for file in list_files_in_s3(S3_BUCKET_NAME, folder)]
        
        if not all_files:
            st.write("No resumes found in the selected folder(s).")
            return
        
        st.write("Resumes found:")
        for folder, file in all_files:
            st.write(f"📁 {folder} - {file}")
        
        if st.button("Process all selected folders"):
            all_candidates = []
            resume_texts = []
            
            for folder, file_name in all_files:
                file_key = f"{folder}/{file_name}"
                st.write(f"Processing: {file_key}")
                
                text = extract_text_with_textract(file_key)
                if not text:
                    continue

                st.write(f"Extracted Text : {text}...")  # Print the first 500 characters
                

                emails, phone_numbers = extract_contact_details(text)  # Extract contact details
                #entities = analyze_text_with_comprehend(text)
                #st.write(f"Extracted Entities: {entities}")
                


                all_candidates.append({
                    "Folder": folder,
                    "Filename": file_name,
                    #"Extracted Text": text[] + "..." if len(text) > 500 else text,
                    "Emails": ", ".join(emails) if emails else "N/A",
                    "Phone Numbers": ", ".join(phone_numbers) if phone_numbers else "N/A",
                    #"Entities": str(entities)
                })
                resume_texts.append(text)
            
            # Compute cosine similarities
            if job_description:
                similarities = compute_cosine_similarity(job_description, resume_texts)
                st.write(f"Cosine Similarities: {similarities}")
                


                # Rank the resumes based on similarity score
                for i, candidate in enumerate(all_candidates):
                    candidate["Similarity"] = similarities[i]
            
            # Sort candidates by similarity (descending)
            sorted_candidates = sorted(all_candidates, key=lambda x: x["Similarity"], reverse=True)
            
            df = pd.DataFrame(sorted_candidates)
            st.table(df)
            
            csv = df.to_csv(index=False)
            st.download_button("Download Results as CSV", csv, "processed_resumes.csv", "text/csv")

if __name__ == "__main__":
    main()
