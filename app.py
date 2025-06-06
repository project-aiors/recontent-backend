from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')
books_df = pd.read_pickle("book_df.pkl")

with open("book_embeddings.pkl", "rb") as f:
    book_embeddings = pickle.load(f)

# Recommendation function
def recommend_books(query, N=15):
    query_embedding = model.encode([query])[0].reshape(1, -1)
    scores = cosine_similarity(query_embedding, book_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:N]
    
    results = []
    for idx in top_indices:
        book = books_df.iloc[idx]
        results.append({
            "bookId": book['bookId'],
            # "title": book['title'],
            # "author": book['author'],
            "score": round(float(scores[idx]), 3)
        })
    return results

# API endpoint
@app.route('/recommend_books', methods=['POST'])
def recommend():
    try:
        data = request.json
        # print('data ',data)
        query = data['query']
        # print('rqt ',query);
        results = recommend_books(query)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run server
if __name__ == '__main__':
    app.run(debug=True)
