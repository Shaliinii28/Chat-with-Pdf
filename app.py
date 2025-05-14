import fitz 
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
import os
import re
import uuid
from flask import Flask, request, render_template, jsonify, session
from werkzeug.utils import secure_filename
from typing import List
import logging

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_pdf_text(pdf_path: str, max_tokens: int = 512, min_chunk_size: int = 50) -> List[str]:
    """Extracts and chunks text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            doc.close()
            raise ValueError("PDF is empty or contains no pages.")
        
        text = ""
        for page in doc:
            page_text = page.get_text("text") or ""
            text += page_text + "\n\n"
        
        doc.close()
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF.")
        
        text = re.sub(r'\n\s*\n+', '\n\n', text.strip())
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        current_token_count = 0
        words_per_token = 0.75
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            word_count = len(para.split())
            token_count = word_count / words_per_token
            
            if token_count > max_tokens:
                words = para.split()
                temp_chunk = ""
                temp_token_count = 0
                
                for word in words:
                    word_token_count = 1 / words_per_token
                    if temp_token_count + word_token_count > max_tokens:
                        if len(temp_chunk) >= min_chunk_size:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                        temp_token_count = word_token_count
                    else:
                        temp_chunk += " " + word
                        temp_token_count += word_token_count
                
                if len(temp_chunk.strip()) >= min_chunk_size:
                    chunks.append(temp_chunk.strip())
                continue
            
            if current_token_count + token_count > max_tokens:
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = para
                current_token_count = token_count
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                current_token_count += token_count
        
        if len(current_chunk.strip()) >= min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else ["No valid chunks created."]
    
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    except Exception as e:
        raise ValueError(f"Error processing PDF: {str(e)}")

def store_chunks_in_chromadb(chunks: List[str], collection_name: str, persist_dir: str = "./chroma_db"):
    """Stores chunks in ChromaDB with embeddings."""
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(name=collection_name)
        
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedder.encode(chunks, show_progress_bar=False)
        
        ids = [str(uuid.uuid4()) for _ in chunks]
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
        return collection
    
    except Exception as e:
        raise ValueError(f"Error storing chunks in ChromaDB: {str(e)}")

def retrieve_relevant_chunks(query: str, collection, top_k: int = 3) -> List[str]:
    """Retrieves relevant chunks from ChromaDB based on query."""
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = embedder.encode([query], show_progress_bar=False)[0]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results['documents'][0] if results['documents'] else []
    
    except Exception as e:
        raise ValueError(f"Error retrieving chunks from ChromaDB: {str(e)}")

def call_gemini_api(prompt: str, context: List[str], api_key: str, model_name: str = "gemini-1.5-flash") -> str:
    """Calls Gemini API with prompt and context."""
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    if not api_key:
        raise ValueError("API key cannot be empty.")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        context_text = "\n\n".join(context) if context else "No context provided."
        full_prompt = f"Context:\n{context_text}\n\nQuestion: {prompt}\nAnswer concisely based on the context."
        
        response = model.generate_content(full_prompt)
        return response.text.strip() if response.text else "No valid response received."
    
    except GoogleAPIError as e:
        raise GoogleAPIError(f"Error calling Gemini API: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

@app.route('/')
def index():
    """Renders the main page."""
    session.pop('chat_history', None)  # Clear chat history on page load
    session.pop('collection_name', None)
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handles PDF upload and processing."""
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file uploaded.'}), 400
        
        file = request.files['pdf']
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'File must be a PDF.'}), 400
        
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_path)
        
        # Process PDF
        chunks = chunk_pdf_text(pdf_path)
        collection_name = f"pdf_{uuid.uuid4().hex}"
        collection = store_chunks_in_chromadb(chunks, collection_name)
        
        # Store collection name in session
        session['collection_name'] = collection_name
        session['chat_history'] = []
        
        # Clean up
        os.remove(pdf_path)
        
        return jsonify({'message': f'PDF processed successfully. Extracted {len(chunks)} chunks.'})
    
    except FileNotFoundError as fnf:
        return jsonify({'error': str(fnf)}), 400
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Unexpected error processing PDF.'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat queries."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty.'}), 400
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({'error': 'Gemini API key not configured.'}), 500
        
        collection_name = session.get('collection_name')
        if not collection_name:
            return jsonify({'error': 'No PDF processed. Please upload a PDF first.'}), 400
        
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name=collection_name)
        
        relevant_chunks = retrieve_relevant_chunks(query, collection)
        if not relevant_chunks:
            response = "No relevant information found in the PDF."
        else:
            response = call_gemini_api(query, relevant_chunks, api_key)
        
        # Update chat history
        chat_history = session.get('chat_history', [])
        chat_history.append({'user': query, 'bot': response})
        session['chat_history'] = chat_history
        
        return jsonify({'response': response, 'chat_history': chat_history})
    
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except GoogleAPIError as gae:
        return jsonify({'error': f"Gemini API error: {str(gae)}"}), 500
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({'error': 'Unexpected error processing query.'}), 500

@app.route('/clear', methods=['POST'])
def clear_chat():
    """Clears the chat history."""
    session.pop('chat_history', None)
    return jsonify({'message': 'Chat history cleared.'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)