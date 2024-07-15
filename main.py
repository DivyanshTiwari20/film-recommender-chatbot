from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the model components
tfidf = joblib.load('tfidf_vectorizer.joblib')
tfidf_matrix = joblib.load('tfidf_matrix.joblib')
movie_data = joblib.load('movie_data.joblib')

def get_movie_info(query):
    query_vec = tfidf.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    indices = similarity.argsort()[-5:][::-1]  # Get top 5 similar movies
    return [movie_data[i] for i in indices]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    results = get_movie_info(user_message)
    response = "Here are some movies that match your query:\n"
    for movie in results:
        response += f"- {movie['Movie Name']} ({movie['Year']}): {movie['Genre']}, {movie['Language']}\n"
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)