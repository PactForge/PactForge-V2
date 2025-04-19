from flask import Flask, request, jsonify, send_from_directory
 import os
 import time
 import chromadb
 from google import genai
 from google.genai import types
 from docx import Document
 from google.api_core import retry
 import spacy
 import logging
 

 app = Flask(__name__, static_folder='frontend')
 

 # Configure logging
 logging.basicConfig(level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s')
 

 # Load the spaCy model for NLP
 nlp = spacy.load("en_core_web_sm")
 

 # Initialize Google API Client with your API key (HARDCODED - UNSAFE)
 GOOGLE_API_KEY = "YOUR_ACTUAL_GOOGLE_API_KEY"  # REPLACE THIS!
 genai.configure(api_key=GOOGLE_API_KEY)
 client = genai.GenerativeModel(model_name="gemini-pro")
 

 # Initialize ChromaDB Client (Adjust for persistence!)
 # This is an in-memory client. For production, use persistent storage.
 clientdb = chromadb.Client()
 collections = {
  "rent": clientdb.get_or_create_collection(name="rent_agreements"),
  "nda": clientdb.get_or_create_collection(name="nda_agreements"),
  "employment": clientdb.get_or_create_collection(name="employ_agreements"),
  "franchise": clientdb.get_or_create_collection(name="franch_agreements"),
  "contract": clientdb.get_or_create_collection(name="contract_agreements"),
 }
 

 # Function to read DOCX files
 def read_docx(endname):
  path = f"sampleagreements/{endname}.docx"
  try:
  doc = Document(path)
  paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
  return paragraphs
  except FileNotFoundError:
  logging.error(f"DOCX file not found: {path}")
  return []
  except Exception as e:
  logging.error(f"Error reading DOCX file: {e}")
  return []
 

 # Function to generate embeddings
 @retry.Retry(predicate=lambda e: isinstance(e, genai.types.GoogleGenerativeAIError) and e.code in {429, 503})
 def generate_embeddings(cl, etype):
  embedding_task = "retrieval_document" if etype else "retrieval_query"
  try:
  embed = client.embed_content(
  model="models/embedding-001",
  contents=cl,
  task_type=embedding_task
  )
  return [e.values for e in embed['embedding'].values]
  except Exception as e:
  logging.error(f"Error generating embeddings: {e}")
  return []
 

 # Function to analyze user input using NLP
 def analyze_input(user_input):
  doc = nlp(user_input)
  keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
  return keywords
 

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
  try:
  response = client.generate_content(
  model="gemini-pro",
  generation_config=types.GenerationConfig(temperature=0.75, top_p=0.9)
  )
  return response.text
  except Exception as e:
  logging.error(f"Error performing analysis: {e}")
  return "Error performing analysis."
 

 @app.route('/')
 def serve_index():
  return send_from_directory('frontend', 'index.html')
 

 @app.route('/generate-agreement', methods=['POST'])
 def generate_agreement():
  data = request.get_json()
  if not data:
  return jsonify({"error": "No data provided"}), 400
  agreement_type = data.get('agreement_type')
  important_info = data.get('important_info')
  extra_info = data.get('extra_info')
 

  if not all([agreement_type, important_info, extra_info]):
  return jsonify({"error": "Missing required fields"}), 400
 

  try:
  # Analyze user input for keywords
  keywords = analyze_input(important_info + " " + extra_info)
 

  # Perform analysis on the provided information
  analysis_result = perform_analysis(agreement_type, important_info)
 

  # Mock response for demonstration
  response = {
  "response": f"Generated {agreement_type} agreement with details: {important_info}, {extra_info}. Analysis result: {analysis_result}. Keywords identified: {keywords}"
  }
 

  return jsonify(response)
  except Exception as e:
  logging.error(f"Error generating agreement: {e}")
  return jsonify({"error": str(e)}), 500
 

 if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)