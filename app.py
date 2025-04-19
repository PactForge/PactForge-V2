from flask import Flask, render_template, request, jsonify
import os
import time
import numpy as np
from google.api_core import retry
import requests
from docx import Document
import chromadb
from chromadb.config import Settings

app = Flask(__name__)

# Configuration
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyAsiNYu4XPPYtco4G7sDsTay_T6tlmptPk')
MODEL_CONFIG = {
    'temperature': 0.75,
    'top_p': 0.9
}

# Initialize ChromaDB client
clientdb = chromadb.Client(Settings(persist_directory="./chroma_db"))
all_dbs = {
    'rent': clientdb.get_or_create_collection(name="rent_agreements"),
    'nda': clientdb.get_or_create_collection(name="nda_agreements"),
    'employment': clientdb.get_or_create_collection(name="employ_agreements"),
    'franchise': clientdb.get_or_create_collection(name="franch_agreements"),
    'contractor': clientdb.get_or_create_collection(name="contract_agreements")
}

# Load sample data
def read_docx(filename, directory):
    path = os.path.join(directory, f"{filename}.docx")
    if not os.path.exists(path):
        # Mock data if file doesn't exist
        return [f"Mock {filename} clause {i}" for i in range(2)]
    doc = Document(path)
    return [p.text for p in doc.paragraphs if p.text.strip()]

def extract_samples(endname):
    dataset_path = f"sampleagreements/{endname}"
    docx_files = []
    if os.path.exists(dataset_path):
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isfile(item_path) and item.lower().endswith('.docx'):
                docx_files.append(item_path)
    return docx_files or [f"Mock {endname} agreement"]

# Load clauses and store in ChromaDB
agreement_types = ['rent', 'nda', 'employment', 'franchise', 'contractor']
all_clauses = {atype: read_docx(atype, 'Clauses') for atype in agreement_types}

for j, atype in enumerate(agreement_types):
    embeds = []
    ids = []
    documents = []
    for i, clause in enumerate(all_clauses[atype]):
        vector = generate_embeddings([clause], True)[0]
        embeds.append(vector)
        ids.append(f"clause-{j}-{i}")
        documents.append(clause)
        time.sleep(0.4)  # Avoid API rate limits
    all_dbs[atype].add(embeddings=embeds, ids=ids, documents=documents)

# Global state
agreement_data = {'agreement_type': '', 'important_info': '', 'extra_info': ''}
step = 1
req = False
final_type = ''
important_info = ''
extra_info = ''
obtained_info = ''

# Retry mechanism
def is_retriable(e):
    return '429' in str(e) or '503' in str(e)

@retry.Retry(predicate=is_retriable)
def generate_content(prompt, model='gemini-2.0-flash'):
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GOOGLE_API_KEY}'
    response = requests.post(url, json={
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': MODEL_CONFIG
    })
    data = response.json()
    if 'error' in data:
        raise Exception(data['error']['message'])
    return data['candidates'][0]['content']['parts'][0]['text']

@retry.Retry(predicate=is_retriable)
def generate_embeddings(cl, etype):
    # Mock embedding generation (replace with Gemini embedContent in production)
    # In production: Use https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent
    task_type = "retrieval_document" if etype else "retrieval_query"
    return [[np.random.rand(768).tolist() for _ in range(len(cl))][0]]

# Utility functions
def query_similar_clauses(query, agreement_type):
    querydb = all_dbs.get(agreement_type)
    if not querydb:
        return []
    query_embed = generate_embeddings([query], False)[0]
    results = querydb.query(query_embeddings=[query_embed], n_results=40)
    return results['documents'][0] if results['documents'] else []

def strip_type(agr):
    agreement_types = ['rent', 'nda', 'contractor', 'employment', 'franchise']
    prompt = f"""Return the type of agreement that the user is referring to in his input "{agr}". Respond in one word, all lowercase. Your responses can only be from the set {agreement_types}. Do not use any punctuation."""
    return generate_content(prompt)

def pos_neg(response):
    prompt = f"""Classify the sentiment of the following sentence. Reply with ONLY '1' if the sentence is positive and ONLY '0' if the sentence is negative.\nSentence = {response}"""
    result = generate_content(prompt)
    return result == '1'

def perform_analysis(atype, impt):
    prompt = f"""You are a legal assistant specializing in determining if the input parameters by the user, defined in "{impt}", are enough parameters to format an agreement of type "{atype}".\nPlease evaluate if the provided information seems to cover all the generally important aspects for a "{atype}" agreement.\nMake sure to evaluate the quality of the input too, if the input seems vague, do consider it as a invalid/bad input.\nYour evaluation must be extremely strict and precise. Any vagueness/lack of information must be considered a severe defect.\nRespond with ONLY these messages, no others:\n"Yes. All essential information seems to be present." if the input appears comprehensive.\n"No, The following essential information seems to be missing or unclear: [list of missing/unclear aspects]" if key details appear to be absent.\n"No, The provided information is too vague or insufficient." if the input is very brief or lacks substantial details."""
    response = generate_content(prompt)
    return response, pos_neg(response)

def obtain_information_holes():
    total_info = important_info + ' ' + extra_info
    prompt = f"""The total information given by the user as an input to generate the agreement of type {final_type} are given in "{total_info}".\nIdentify any missing or unclear information needed to generate a {final_type} agreement based on the provided user input: "{total_info}". As a comprehensive legal assistant, pinpoint specific details that require clarification or are absent from the input.\nGenerate your final prompt in a way such that if it is passed into a Google search, it gives back the required information."""
    return generate_content(prompt)

def get_data(holes):
    prompt = f"""As an LLM, you have identified a few information deficiencies, outlined in "{holes}", required to generate a LAW agreement of type {final_type}.\nYou are supposed to retrieve the relevant information using Google search. Make sure to keep it concise and accurate."""
    return generate_content(prompt)

def generate_agreement():
    relevant_clauses = query_similar_clauses(extra_info, final_type)
    sample_agreement_paths = extract_samples(final_type)
    prompt = f"""You are a helpful AI assistant for law agreement generation. The names, dates, locations, and information are stored in "{important_info}".\nThe agreement type is "{final_type}".\n{relevant_clauses} contains 40 most common used clauses in the current agreements, with relevance sorted from highest to lowest, depending on this current use case. Make sure to read through them, understand them, and use the most relevant documents according to the user's wish as outlined in "{extra_info}".\nMake sure your clauses are end-to-end, non-manipulatable, unable to have loopholes, and concise and readable. While referring to government officers, refer to them as specifically as possible to avoid confusion.\nA few example agreement formats are outlined in "{sample_agreement_paths}". Structure them similarly and provide a concise output. The English must be clean, non-confusing, and clear enough to be understood by a common man, but complex enough to uphold legal intricacies and important points.\nSome important information that has been obtained through Google search is given in "{obtained_info}". Use it in your generation as requested by the user.\nFormat a full {final_type} agreement as provided in the samples and give the final output.\n\nRelevant Clauses: {', '.join(relevant_clauses)}"""
    return generate_content(prompt)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    global step, agreement_data, req, final_type, important_info, extra_info, obtained_info
    user_input = request.json.get('message', '').strip()
    if not user_input:
        return jsonify({'response': '', 'step': step})

    try:
        if step == 1:
            agreement_data['agreement_type'] = user_input
            final_type = strip_type(user_input)
            response = f"Excellent choice! For the {final_type} agreement, please provide critical details (e.g., parties involved, duration, financial terms)."
            step = 2
        elif step == 2:
            agreement_data['important_info'] = user_input
            important_info = user_input
            response = f"Thank you! Any specific clauses or custom details for the {final_type} agreement?"
            step = 3
        else:
            agreement_data['extra_info'] = user_input
            extra_info = user_input
            analysis_response, is_sufficient = perform_analysis(final_type, important_info)
            if not is_sufficient:
                step = 2
                response = analysis_response + "\nPlease provide more details as prompted."
            else:
                info_holes = obtain_information_holes()
                obtained_info = get_data(info_holes)
                response = generate_agreement()
                step = 1
                agreement_data = {'agreement_type': '', 'important_info': '', 'extra_info': ''}
                final_type = ''
                important_info = ''
                extra_info = ''
                obtained_info = ''
                req = False
        return jsonify({'response': response, 'step': step})
    except Exception as e:
        return jsonify({'response': f"Error: {str(e)}", 'step': step})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))