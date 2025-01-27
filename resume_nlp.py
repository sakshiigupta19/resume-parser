import streamlit as st
import pdfplumber
import re
import csv
import hashlib
import io
from datetime import datetime

# Functions from your previous code
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
        skills = re.split(r"[\u2022\n,]+", skills_text)  # Bullet operators unicode
        return clean_skills([skill.strip() for skill in skills if skill.strip()])
    return []

def clean_skills(raw_skills):
    """Clean and normalize extracted skills."""
    cleaned = []
    for skill in raw_skills:
        cleaned_skill = re.sub(r"^\s*-?\s*", "", skill)
        cleaned_skill = re.sub(r"\(.*?\)", "", cleaned_skill).strip()  # Text enclosed in parentheses
        words = re.split(r"[,\s:/;]+", cleaned_skill)
        cleaned.extend(word.strip() for word in words if word.strip())
    # Remove duplicates and normalize case
    return list(dict.fromkeys(word.lower() for word in cleaned))

def extract_experience(text):
    """Extracts work experience (start and end dates) from the resume."""
    # Regex to find work experience dates (assuming formats like 'Jan 2020 - Dec 2022')
    experience_matches = re.findall(r"([A-Za-z]+ \d{4})\s*[-to–]\s*([A-Za-z]+ \d{4})", text)
    experience_years = []
    for start, end in experience_matches:
        try:
            start_date = datetime.strptime(start, "%b %Y")
            end_date = datetime.strptime(end, "%b %Y")
            experience_years.append((start_date, end_date))
        except ValueError:
            continue
    return experience_years

def calculate_experience(experience_years):
    """Calculates the total number of years of experience based on the extracted work dates."""
    total_years = 0
    for start_date, end_date in experience_years:
        # Calculate experience duration in years
        duration = (end_date - start_date).days / 365.25  # Average year length considering leap years
        total_years += duration
    return total_years

def extract_info(text):
    """Extracts key information (email, phone, skills, and experience)."""
    data = {
        "email": re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text),
        "phone": re.search(r"(\+?\d{1,4}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}", text),
        "skills": extract_skills(text),
        "experience": extract_experience(text),
    }
    result_data = {}
    for key, value in data.items():
        if key != "skills" and key != "experience":
            result_data[key] = value.group(0) if value else "Not available"
        elif key == "skills":
            result_data[key] = value
        elif key == "experience":
            result_data[key] = value
    return result_data

def evaluate_skills(candidate_skills, required_skills):
    """Matches candidate's skills with required skills."""
    candidate_set = set(map(str.lower, candidate_skills))
    required_set = set(map(str.lower, required_skills.split(", ")))
    matched_skills = candidate_set.intersection(required_set)
    match_percentage = (len(matched_skills) / len(required_set)) * 100 if required_set else 0
    return matched_skills, match_percentage

def process_single_pdf(file, required_skills, processed_hashes):
    """Processes a single PDF resume file and prevents duplicate processing."""
    file_hash = generate_file_hash(file)
    
    # Skip file if it's already processed
    if file_hash in processed_hashes:
        return None
    
    text = load_pdf(file)
    data = extract_info(text)
    
    # Calculate experience in years
    experience_years = calculate_experience(data["experience"])
    data["experience_years"] = experience_years

    matched_skills, match_percentage = evaluate_skills(data["skills"], required_skills)

    candidate_data = data.copy()
    candidate_data["matched_skills"] = ", ".join(matched_skills)
    candidate_data["match_percentage"] = match_percentage
    processed_hashes.add(file_hash)  # Add file hash to processed set

    return candidate_data

def generate_file_hash(file):
    """Generates a unique hash for the uploaded file based on its content."""
    file_content = file.read()  # Read the file content
    file.seek(0)  # Reset file pointer after reading
    return hashlib.md5(file_content).hexdigest()  # Generate MD5 hash of the content

def save_to_csv(data, file_name):
    """Saves extracted data to CSV."""
    with open(file_name, mode='a', newline='', encoding='utf-8', errors='ignore') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Check if the file is empty to write headers
            writer.writerow(data.keys())  # Write headers
        writer.writerow(data.values())  # Write values

# Streamlit UI
def main():
    st.title("Resume Scanner and Matcher")
    st.subheader("Upload resumes (PDF format) to extract information and evaluate skills match")

    # User-friendly skill input options
    predefined_skills = [
        "Python", "SQL", "React", "Django", "Java", "JavaScript", "HTML", "CSS", "Node.js", "AWS", "Git", "Machine Learning", "Data Analysis"
    ]
    selected_skills = st.multiselect(
        "Select required skills from the list:",
        predefined_skills
    )
    
    if not selected_skills:
        required_skills = st.text_area("Or, enter Required Skills (comma-separated)")
        if not required_skills:
            st.warning("Please select or enter at least one required skill.")
    else:
        required_skills = ", ".join(selected_skills)
    
    uploaded_files = st.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)
    processed_hashes = set()
    
    if uploaded_files:
        st.write(f"Processing {len(uploaded_files)} resumes...")

        # Process the uploaded files
        all_candidates = []
        for uploaded_file in uploaded_files:
            candidate_data = process_single_pdf(uploaded_file, required_skills, processed_hashes)
            # Skip processing if the file was already processed
            if candidate_data is None:
                st.write(f"Skipping duplicate file: {uploaded_file.name}")
                continue
            candidate_data["filename"] = uploaded_file.name
            all_candidates.append(candidate_data)

        # Rank candidates based on match percentage
        ranked_candidates = sorted(all_candidates, key=lambda x: x["match_percentage"], reverse=True)

        # Show ranked candidates in Streamlit as a table
        if ranked_candidates:
            st.write("**Ranked Candidates:**")
            
            # Prepare a list of dictionaries for displaying in table format
            table_data = []
            for candidate in ranked_candidates:
                table_data.append({
                    "Filename": candidate["filename"],
                    "Email": candidate["email"],
                    "Phone": candidate["phone"],
                    "Matched Skills": candidate["matched_skills"],
                    "Experience (Years)": f"{candidate['experience_years']:.2f}",
                    "Match Percentage": f"{candidate['match_percentage']:.2f}%",
                })
            
            # Display the table
            st.table(table_data)
        else:
            st.warning("No valid candidates found.")

        # Option to download the CSV of ranked candidates
        if st.button("Download Ranked Candidates CSV"):
            output_csv = "ranked_resumes.csv"
            with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["email", "phone", "match_percentage", "matched_skills", "experience_years", "skills", "filename"])  # CSV header
                for candidate in ranked_candidates:
                    writer.writerow([  
                        candidate["email"],
                        candidate["phone"],
                        candidate["match_percentage"],
                        candidate["matched_skills"],
                        f"{candidate['experience_years']:.2f}",
                        ", ".join(candidate["skills"]),
                        candidate["filename"]
                    ])
            st.write(f"CSV file saved as {output_csv}. You can download it here:")
            st.download_button("Download CSV", data=open(output_csv, "rb"), file_name=output_csv)

if __name__ == "__main__":
    main()
