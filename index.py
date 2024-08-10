import uuid
from flask import Flask, request, jsonify, render_template, redirect, session
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

app = Flask(__name__)

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("my_collection")

model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/')
def index():
    success_message = None
    if 'embedding_added' in session:
        success_message = 'Embedding added successfully!'
        session.pop('embedding_added')
    search_results = session.get('search_results', [])
    if search_results:
        session.pop('search_results')
    return render_template('embeddings.html', success_message=success_message, search_results=search_results)

@app.route('/store', methods=['POST'])
def store_embedding():
    try:
        text_embedding = request.form['embedding']
        metadata = dict(metadata=request.form.get('metadata', {}))
        embedding = model.encode(text_embedding)
        embedding = embedding.tolist()
        chroma_collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            metadatas=[metadata]
        )
        session['embedding_added'] = True
        return redirect('/')
    except Exception as e:
        print(f'Error storing embedding: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search_embeddings():
    query = request.form['query_embedding']
    query_embedding = model.encode([query])
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    if results:
        session['search_results'] = results
        return redirect('/')
    else:
        return jsonify({'error': 'No results found'}), 404

if __name__ == '__main__':
    app.run(debug=True)