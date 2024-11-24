import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are available
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load the CSV file containing articles
csv_file = "articles.csv"  # Update the path as necessary
df = pd.read_csv(csv_file)

# Ensure the CSV has the expected structure
if "content" not in df.columns:
    raise KeyError(f"'content' column not found in {csv_file}. Ensure the CSV has 'content' as a column.")

# Manual tokenization function
def tokenize(text):
    """
    Tokenizes text by splitting it manually and removing extra whitespace.
    Handles edge cases with punctuation and special characters.
    """
    # Replace non-alphabetic characters with spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Split text into words based on whitespace
    tokens = text.split()
    return tokens

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize manually
    tokens = tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Preprocess all articles
df['processed_content'] = df['content'].apply(preprocess_text)

# Combine all processed text for TF-IDF analysis
corpus = df['processed_content'].tolist()

# Use TF-IDF to identify the most important terms
tfidf_vectorizer = TfidfVectorizer(max_features=50)  # Adjust max_features as needed
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
keywords = tfidf_vectorizer.get_feature_names_out()

# Initialize the document-term matrix
dtm = []

# Generate the document-term matrix
for text in df['processed_content']:
    word_count = {word: 0 for word in keywords}
    for word in text.split():
        if word in word_count:
            word_count[word] += 1
    dtm.append([word_count[keyword] for keyword in keywords])

# Convert DTM to a DataFrame
dtm_df = pd.DataFrame(dtm, columns=keywords)

# Add document names or indices as the first column
dtm_df.insert(0, 'Document', df.index)

# Save the DTM to a CSV file
output_csv = "optimized_document_term_matrix.csv"
dtm_df.to_csv(output_csv, index=False)

print(f"Optimized Document-Term Matrix saved to {output_csv}")
