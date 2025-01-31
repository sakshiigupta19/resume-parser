
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample texts
text1 = "Software engineer with experience in Python and machine learning."
#text2 = "Looking for a software engineer with expertise in Python and AI."
text2="Software engineer with experience in Python and machine learning."

vectorizer = TfidfVectorizer()

# Fit and transform the texts into TF-IDF matrices
tfidf_matrix = vectorizer.fit_transform([text1, text2])

# cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

# Print the cosine similarity score
print(f"Cosine Similarity: {cosine_sim[0][0]}")
"""


import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Preprocessing Function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

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

# Example Usage
if __name__ == "__main__":
    job_desc = "Looking for a data scientist with experience in machine learning and NLP."
    resumes = [
        "Experienced data scientist skilled in machine learning and deep learning.",
        "Software engineer with knowledge of Python and cloud computing.",
        "Machine learning expert proficient in NLP and data analytics."
    ]
    ranked_resumes = rank_resumes(job_desc, resumes)
    for idx, (resume, score) in enumerate(ranked_resumes, start=1):
        print(f"Rank {idx}: {resume} (Score: {score:.2f})")
