import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pandas as pd 
import docx

# Define the reference documents

request = requests.get('')

references = [request]

# Define the text to be checked for plagiarism

target_text = docx.Document('text.docx').paragraphs[0].text

# Preprocess the text by removing stop words and punctuation
stop_words = set(stopwords.words('english'))
tokens = word_tokenize(target_text)
filtered_tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

# Compute the similarity between the text and the reference documents using TF-IDF and cosine similarity
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(references + [target_text])
similarity_scores = cosine_similarity(vectors)[-1][:-1]

# Set a threshold for the similarity score
threshold = 0.9

# Check if the similarity score exceeds the threshold
if max(similarity_scores) > threshold:
    print('The text is plagiarized.')
else:
    print('The text is not plagiarized.')