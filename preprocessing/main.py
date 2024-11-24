import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are available
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Predefined list of keywords (categories)
keywords = ["algorithm", "cellular", "game", "hardware", "internet", 
            "mobile", "network", "search", "secure", "web"]

# Load the CSV file containing articles
csv_file = "articles.csv"  # Update the path as necessary
df = pd.read_csv(csv_file)

# Ensure the CSV has the expected structure
if "content" not in df.columns:
    raise KeyError(f"'content' column not found in {csv_file}. Ensure the CSV has 'content' as a column.")

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return []
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, URLs, and numbers
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Initialize an empty document-term matrix
dtm = []

# Process each document
for index, row in df.iterrows():
    text = row['content']
    tokens = preprocess_text(text)
    # Count keyword frequencies
    word_count = Counter(tokens)
    row_data = [word_count.get(keyword, 0) for keyword in keywords]
    dtm.append(row_data)

# Convert DTM to a DataFrame
dtm_df = pd.DataFrame(dtm, columns=keywords)

# Add document names or indices as the first column
dtm_df.insert(0, 'Document', df.index)

# Save the DTM to a CSV file
output_csv = "document_term_matrix.csv"
dtm_df.to_csv(output_csv, index=False)

print(f"Document-Term Matrix saved to {output_csv}")
