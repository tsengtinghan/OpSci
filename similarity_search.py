from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from preprocessing import preprocess_text_without_lemmatization

def compute_similarities(user_input, df, column_to_search='refined_description'):
    """Compute cosine similarities between user input and specified column in dataframe."""
    # Preprocess user input
    processed_user_input = preprocess_text_without_lemmatization(user_input)
    
    # Initialize a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    
    # Fit and transform the specified column
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column_to_search])
    
    # Transform the user input using the TF-IDF vectorizer
    user_input_vector = tfidf_vectorizer.transform([processed_user_input])

    # Compute the cosine similarity between the user's input and the specified column
    return cosine_similarity(user_input_vector, tfidf_matrix)

def get_top_n_projects(n, cosine_similarities, df):
    """Retrieve the top N projects based on cosine similarities."""
    # Get the indices of the top N projects by similarity
    top_indices = cosine_similarities.argsort().flatten()[-n:]

    # Retrieve the top N project URLs
    # projects = [{"name": df.iloc[index]["project_name"], "url": df.iloc[index]["project_url_on_catalog"]} for index in top_indices]
    # Retrieve the top 5 project names, URLs, and refined descriptions
    projects = [{"name": df.iloc[index]["project_name"], "url": df.iloc[index]["project_url_on_catalog"], "refined_description": df.iloc[index]["refined_description"]} for index in top_indices]

    return projects
