from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/embed', methods=['POST'])
def embed():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "Missing text"}), 400

    embedding = model.encode(text).tolist()
    return jsonify({"embedding": embedding})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

