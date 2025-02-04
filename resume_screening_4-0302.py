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

from fuzzywuzzy import fuzz
from fuzzywuzzy import process


SKILLS_LIST = [
    'agile', 'android', 'api', 'angularjs', 'awk', 'c', 'c++', 'cloud computing', 'css', 
    'data analysis', 'data structures', 'debugging', 'deep learning', 'django', 'docker', 
    'eclipse', 'excel', 'flask', 'git', 'google cloud', 'hadoop mapreduce', 'html', 'html5', 
    'java', 'javascript', 'jsp', 'jsf', 'jupyter', 'kubernetes', 'latex', 'linux', 'machine learning', 
    'matlab', 'maven', 'model optimization', 'mysql', 'natural language processing', 'neural networks', 
    'perl', 'php', 'postgresql', 'project management', 'python', 'pytorch', 'rest apis', 'ruby', 
    'scheme', 'scikit-learn', 'scrum', 'sed', 'shell', 'sql', 'spring mvc', 'text analysis', 
    'tensorflow', 'vim', 'windows', 'xml'
]



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

def compute_skill_similarity(job_skills, resume_skills):
    """Compute similarity score based on matching skills."""
    if not job_skills or not resume_skills:
        return 0.0
    
    # Find the common skills between job description and resume
    common_skills = set(job_skills) & set(resume_skills)
    similarity_score = len(common_skills) / len(set(job_skills))  # Normalize by the number of skills in the job description
    return similarity_score


nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w.lower()) for w in words if w.lower() not in stop_words]
    return " ".join(cleaned_words)


# Function to extract skills with fuzzy matching and whole word matching
def extract_skills_from_text(text, job_skills, SKILLS_LIST):
    """Extract skills from text by matching with predefined skills list using whole word matching and fuzzy matching."""
    
    # Convert the text to lowercase for consistency and case-insensitivity
    text = text.lower()

    # Compile regex patterns for skills from the job_skills list and the SKILLS_LIST
    job_skill_patterns = {skill: re.compile(r'\b' + re.escape(skill.lower()) + r'\b') for skill in job_skills}
    all_skill_patterns = {skill: re.compile(r'\b' + re.escape(skill.lower()) + r'\b') for skill in SKILLS_LIST}

    matched_skills = []
    all_resume_skills = []

    # First, find exact matches from job_skills using regex
    for skill, pattern in job_skill_patterns.items():
        if pattern.search(text):
            matched_skills.append(skill)

    # Next, find exact matches for skills in the SKILLS_LIST
    for skill, pattern in all_skill_patterns.items():
        if pattern.search(text):
            all_resume_skills.append(skill)

    # Fuzzy matching: Handle cases where skills might be misspelled or slightly modified
    for word in set(text.split()):  # We split the text into unique words
        if word not in matched_skills:  # Avoid checking already matched words
            closest_matches = process.extract(word, SKILLS_LIST, limit=5, scorer=fuzz.ratio)
            for match, score in closest_matches:
                if score >=80:  # Only add to matches if the fuzzy match score exceeds the threshold
                    if match not in all_resume_skills:
                        all_resume_skills.append(match)
                    if match in job_skills and match not in matched_skills:
                        matched_skills.append(match)

    return matched_skills, all_resume_skills


def main():
    st.title("Carnera Resume Screening")
    st.subheader("Extract text and analyze entities from resumes in S3")
    
    job_description = st.text_area("Enter Job Description:", "").lower()
    st.write(f"Processed Job Description: {job_description}")
    # Preprocess the job description
    job_description_preprocessed = preprocess_text(job_description)
    

    # Extract skills from the job description
    job_skills, _ = extract_skills_from_text(job_description,SKILLS_LIST,SKILLS_LIST)
    st.write(f"Skills in Job Description: {', '.join(job_skills)}")

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

                #st.write(f"Extracted Text : {text}...")  # Print the first 500 characters
                

                emails, phone_numbers = extract_contact_details(text)  # Extract contact details
                #entities = analyze_text_with_comprehend(text)
                #st.write(f"Extracted Entities: {entities}")
                
                # Extract skills from resume
                resume_skills, all_resume_skills = extract_skills_from_text(text,job_skills,SKILLS_LIST)
                matched_skills_str = ", ".join(resume_skills) if resume_skills else "No skills matched"
                #all_skills_str = ", ".join(all_resume_skills) if all_resume_skills else "No skills found"
                # Calculate remaining skills (those that are in the resume but not matched with the job skills)
                remaining_skills = ",".join([skill for skill in all_resume_skills if skill not in resume_skills])
        
                preprocessed_resume = preprocess_text(text)

                # Compute skill similarity (for matched skills)
                skill_similarity = compute_skill_similarity(job_skills, resume_skills)


                all_candidates.append({
                    #"Folder": folder,
                    "Filename": file_name,
                    #"Preprocessed Resume Text": preprocessed_resume[:300],
                    "Emails": ", ".join(emails) if emails else "N/A",
                    "Phone Numbers": ", ".join(phone_numbers) if phone_numbers else "N/A",
                    #"Entities": str(entities)
                    "Matched Skills": matched_skills_str,
                    "Remaining Skills from Resume": remaining_skills,
                    "Matched Skill Similarity": skill_similarity,
                })
                resume_texts.append(text)
            
            # Compute cosine similarities for resumes
            if job_description:
                similarities = compute_cosine_similarity(job_description_preprocessed, resume_texts)
                
                # Rank the resumes based on similarity score
                for i, candidate in enumerate(all_candidates):
                    candidate["Cosine Similarity"] = similarities[i]
                    
            # Sort candidates by similarity (descending)
            sorted_candidates = sorted(all_candidates, key=lambda x: x["Cosine Similarity"], reverse=True)
            
            df = pd.DataFrame(sorted_candidates)
            st.table(df)
            
            csv = df.to_csv(index=False)
            st.download_button("Download Results as CSV", csv, "processed_resumes.csv", "text/csv")

if __name__ == "__main__":
    main()
