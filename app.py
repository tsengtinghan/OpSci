import openai
from flask import Flask, render_template, request, jsonify
import pandas as pd
from preprocessing import preprocess_text_without_lemmatization
from similarity_search import compute_similarities, get_top_n_projects

app = Flask(__name__)

# Load the CSV file and preprocess project descriptions
df = pd.read_csv('feed.csv')
df['processed_description'] = df['project_description'].apply(preprocess_text_without_lemmatization)

# Create a combined column for keywords and fields_of_science
df['keywords_and_fields'] = df['keywords'].fillna('') + ' ' + df['fields_of_science'].fillna('')
df['processed_keywords_and_fields'] = df['keywords_and_fields'].apply(preprocess_text_without_lemmatization)

@app.route("/")
def index():
    return render_template("chat_interface_advanced_styled.html")

@app.route("/get_matches", methods=["POST"])
def get_matches():
    user_input = request.json["user_input"]
    search_type = request.json["search_type"]
    threshold = float(request.json["threshold"])
    
    # Choose the column to search based on user's selection
    column_to_search = 'refined_description' if search_type == 'description' else 'processed_keywords_and_fields'
    
    cosine_similarities = compute_similarities(user_input, df, column_to_search)
    top_5_indices = cosine_similarities.argsort().flatten()[-5:]
    
    # print(cosine_similarities.flatten())
    # above_threshold_indices = [i for i, score in enumerate(cosine_similarities.flatten()) if score > threshold]
    # sorted_indices = sorted(above_threshold_indices, key=lambda x: cosine_similarities[x], reverse=True)
    # top_5_indices = sorted_indices[:5]
    

    # Retrieve the top 5 project names, URLs, and refined descriptions
    projects = [
        {
            "name": df.iloc[index]["project_name"],
            "url": df.iloc[index]["project_url_on_catalog"],
            "refined_description": df.iloc[index]["refined_description"]
        } 
        for index in top_5_indices
    ]
    
    project = projects.reverse()
    
    

    response = jsonify(projects=projects)
    response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0')  # Disable caching
    return response

@app.route("/get_advanced_matches", methods=["POST"])
def get_advanced_matches():
    data = request.json
    description = data["description"]
    # For now, just use the description for similarity search
    cosine_similarities = compute_similarities(description, df, 'refined_description')
    top_5_indices = cosine_similarities.argsort().flatten()[-5:]
    
    # Retrieve the top 5 project names, URLs, and refined descriptions
    projects = [
        {
            "name": df.iloc[index]["project_name"],
            "url": df.iloc[index]["project_url_on_catalog"],
            "refined_description": df.iloc[index]["refined_description"]
        } 
        for index in top_5_indices
    ]
    
    response = jsonify(projects=projects)
    response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0')  # Disable caching
    return response

if __name__ == "__main__":
    app.run(debug=True)
