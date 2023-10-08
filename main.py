
import pandas as pd
from preprocessing import preprocess_text_without_lemmatization
from similarity_search import compute_similarities, get_top_n_projects

def main():
    # Load the CSV file
    df = pd.read_csv('feed.csv')
    
    # Preprocess the project descriptions
    df['processed_description'] = df['project_description'].apply(preprocess_text_without_lemmatization)
    
    # Sample user input
    user_input = "I'm interested in space exploration and NASA projects."
    
    # Compute cosine similarities
    cosine_similarities = compute_similarities(user_input, df)
    
    # Retrieve the top 5 project URLs
    top_5_urls = get_top_n_projects(5, cosine_similarities, df)
    
    print("Top 5 similar project URLs:")
    for url in top_5_urls:
        print(url)

if __name__ == "__main__":
    main()

