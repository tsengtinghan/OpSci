
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_text_without_lemmatization(text):
    # Preprocess text by converting to lowercase, removing punctuation and stopwords.
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize and remove stopwords
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)


