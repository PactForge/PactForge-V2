from flask import Flask, request, jsonify
import os
from google import genai
from google.genai import types
from google.api_core import retry
from flask_cors import CORS
from docx import Document
import chromadb
import numpy as np
import re

app = Flask(__name__)
CORS(app)  # Enable CORS

# --- Gemini API Setup ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)
model_config = types.GenerateContentConfig(
    temperature=0.75,
    top_p=0.9,
)
generation_model = genai.GenerativeModel(model_name="gemini-pro", generation_config=model_config)

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

# --- Language-Specific NLP and Embedding Model (Placeholder -  ***CRITICAL TO IMPLEMENT*** ) ---
# from sentence_transformers import SentenceTransformer
# embedding_model = SentenceTransformer('your-multilingual-model')

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords_and_intent(user_input_language):
    # *** REPLACE THIS PLACEHOLDER WITH YOUR LANGUAGE-SPECIFIC NLP ***
    keywords = [word for word in clean_text(user_input_language).lower().split() if len(word) > 2]
    intent = "franchise_agreement"  # Improve intent detection
    return keywords, intent

def generate_embeddings_language(text_list):
    # *** REPLACE THIS PLACEHOLDER WITH YOUR EMBEDDING GENERATION ***
    # embeddings = embedding_model.encode(text_list)
    return [np.random.rand(768).tolist() for _ in text_list]  # Dummy embeddings

# --- ChromaDB Setup ---
CHROMA_CLIENT = chromadb.HttpClient(host="your_chromadb_host", port="your_chromadb_port")  # Replace
CLAUSE_COLLECTION_NAME = "legal_clauses_franchise"
clause_collection = CHROMA_CLIENT.get_or_create_collection(name=CLAUSE_COLLECTION_NAME)

def load_clauses_into_chroma(docx_file_path):
    try:
        doc = Document(docx_file_path)
        clauses = [clean_text(p.text) for p in doc.paragraphs if clean_text(p.text)]
        embeddings = generate_embeddings_language(clauses)
        ids = [f"clause-{i}" for i in range(len(clauses))]
        clause_collection.add(embeddings=embeddings, ids=ids, documents=clauses)
        print(f"Successfully loaded clauses from {docx_file_path} into ChromaDB.")
    except Exception as e:
        print(f"Error loading clauses from {docx_file_path}: {e}")

# Load clauses (ONE-TIME SETUP)
# Example:
# load_clauses_into_chroma("franchise.docx")

# --- Clause Retrieval ---
def retrieve_relevant_clauses(user_input_language, top_n=5):
    try:
        input_embedding = generate_embeddings_language([user_input_language])[0]
        results = clause_collection.query(
            query_embeddings=[input_embedding],
            n_results=top_n
        )
        return results['documents']
    except Exception as e:
        print(f"Error retrieving clauses: {e}")
        return []

# --- Agreement Template (Basic Example - Expand) ---
agreement_templates = {
    "franchise_agreement": """
    FRANCHISE AGREEMENT

    [INTRODUCTION]

    1.  GRANT OF FRANCHISE: [GRANT_CLAUSE]

    2.  TERRITORY: [TERRITORY_CLAUSE]

    ...

    [OTHER_CLAUSES]

    [CONCLUSION]
    """
}

def fill_agreement_template(template, clauses):
    filled_template = template
    filled_template = filled_template.replace("[GRANT_CLAUSE]", clauses[0] if len(clauses) > 0 else "")
    filled_template = filled_template.replace("[TERRITORY_CLAUSE]", clauses[1] if len(clauses) > 1 else "")
    filled_template = filled_template.replace("[OTHER_CLAUSES]", "\n".join(clauses[2:]))
    return filled_template

# --- Main Agreement Generation ---
@app.route('/generate-agreement', methods=['POST'])
def handle_generation():
    data = request.get_json()
    user_input_language = data.get('user_input', '')

    if not user_input_language:
        return jsonify({'response': "Please provide your request."})

    try:
        keywords, intent = extract_keywords_and_intent(user_input_language)
        relevant_clauses = retrieve_relevant_clauses(user_input_language)

        agreement_template = agreement_templates.get(intent, agreement_templates["franchise_agreement"])
        filled_agreement = fill_agreement_template(agreement_template, relevant_clauses)

        prompt = f"""You are a helpful AI assistant for generating franchise agreements.
        The user's request is: {user_input_language}
        Here are relevant clauses: {relevant_clauses}
        Here is a template: {agreement_template}

        Generate a complete and legally sound franchise agreement, incorporating the clauses into the template.
        """

        response = generation_model.generate_content(prompt)
        generated_text = response.text + "\n\n" + filled_agreement
        return jsonify({'response': generated_text})

    except Exception as e:
        return jsonify({'response': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)