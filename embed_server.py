from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

print("Loading model...")  # Add this
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")


@app.route('/embed', methods=['POST'])
def embed():
    print("Received embedding request")
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "Missing text"}), 400

    embedding = model.encode(text).tolist()
    return jsonify({"embedding": embedding})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

