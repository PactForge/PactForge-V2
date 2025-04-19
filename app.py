from flask import Flask, request, jsonify, send_from_directory
import os
import time
import chromadb
from google import genai
from google.genai import types
from docx import Document

app = Flask(__name__, static_folder='frontend')

# Initialize Google API Client with your API key
GOOGLE_API_KEY = "AIzaSyAlZXQHr4PN5Po0KeNPGfYFjuYaI3jMcB0"
client = genai.Client(api_key=GOOGLE_API_KEY)

# Initialize ChromaDB Client
clientdb = chromadb.Client()
# Create or get collections
collections = {
    "rent": clientdb.get_or_create_collection(name="rent_agreements"),
    "nda": clientdb.get_or_create_collection(name="nda_agreements"),
    "employment": clientdb.get_or_create_collection(name="employ_agreements"),
    "franchise": clientdb.get_or_create_collection(name="franch_agreements"),
    "contract": clientdb.get_or_create_collection(name="contract_agreements"),
}

# Function to read DOCX files
def read_docx(endname):
    path = f"/kaggle/input/agreement-clauses-capstone/{endname}.docx"  # Updated path
    doc = Document(path)
    paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
    return paragraphs

# Function to generate embeddings
def generate_embeddings(cl, etype):
    if etype:
        embedding_task = "retrieval_document"
    else:
        embedding_task = "retrieval_query"
    embed = client.models.embed_content(
        model="models/text-embedding-004",
        contents=cl,
        config=types.EmbedContentConfig(task_type=embedding_task)
    )
    return [e.values for e in embed.embeddings]

# Function to perform analysis on user input
def perform_analysis(agreement_type, important_info):
    prompt = f"""You are a legal assistant specializing in determining if the input parameters by the user, defined in {important_info} are enough parameters to format an agreement of type {agreement_type}. 
    Please evaluate if the provided information seems to cover all the generally important aspects for a '{agreement_type}' agreement.
    Your evaluation must be extremely strict and precise. Any vagueness/lack of information must be considered a severe defect.
    Respond with ONLY these messages, no others.
    - "Yes. All essential information seems to be present." if the input appears comprehensive.
    - "No, The following essential information seems to be missing or unclear: [list of missing/unclear aspects]" if key details appear to be absent.
    - "No, The provided information is too vague or insufficient." if the input is very brief or lacks substantial details.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.75, top_p=0.9)
    )
    return response.text

# Endpoint to serve the index.html
@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')

# Endpoint to generate agreement
@app.route('/generate-agreement', methods=['POST'])
def generate_agreement():
    data = request.json
    agreement_type = data['agreement_type']
    important_info = data['important_info']
    extra_info = data['extra_info']

    # Perform analysis on the provided information
    analysis_result = perform_analysis(agreement_type, important_info)

    # Mock response for demonstration
    response = {
        "response": f"Generated {agreement_type} agreement with details: {important_info}, {extra_info}. Analysis result: {analysis_result}"
    }

    # Here you would implement the logic to generate the actual agreement
    # For example, using the provided information to create a legal document

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)