import uuid
from flask import Flask, request, jsonify, render_template, redirect, session
import chromadb
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("my_collection")

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
        chroma_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text_embedding],
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
    results = chroma_collection.query(
        query_texts=[query],
        n_results=2
    )
    if results:
        print(f'Search results: {results}')
        formatted_results = []
        for x in range(len(results['documents'][0])):
            formatted_results.append({
                "document": results['documents'][0][x],
                "distance": results['distances'][0][x]
            })
        session['search_results'] = formatted_results
        return redirect('/')
    else:
        return jsonify({'error': 'No results found'}), 404

if __name__ == '__main__':
    app.run(debug=True)