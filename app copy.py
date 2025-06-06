from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models and data
movies_dict = pickle.load(open('movie_details_cleaned.pkl', 'rb'))
df = pd.DataFrame(movies_dict)
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
count_matrix = pickle.load(open('count_matrix.pkl', 'rb'))

# Recommendation function
def recommend_movies_knn1(title, df=df, model=knn_model):
    idx = df[df['id'] == title].index[0]
    distances, indices = model.kneighbors(count_matrix[idx], n_neighbors=13)

    similar_movies_indices = indices.flatten()[1:]
    return df['id'].iloc[similar_movies_indices].tolist()

# API route for getting recommendations by movie ID
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        movie_id = data['movie_id']
        recommendations_knn = recommend_movies_knn1(movie_id)
        return jsonify({"recommended_movie_ids": recommendations_knn})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run()
