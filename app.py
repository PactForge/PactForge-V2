from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
from google.api_core import retry
import time
import chromadb
from docx import Document
import google.generativeai as genai
from google.generativeai import types

app = Flask(__name__)
CORS(app)

# Hardcoded Google API key (not recommended for production)
GOOGLE_API_KEY = "AIzaSyAQeDHegBbQZ_0-oGS_KxvBPrGvU9Nfcns"

client = genai.Client(api_key=GOOGLE_API_KEY)

model_config = types.GenerateContentConfig(
    temperature=0.75,
    top_p=0.9,
)

def read_docx(endname):
    path = f"/app/data/clauses/{endname}.docx"  # Updated path for Render
    doc = Document(path)
    paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
    return paragraphs

def extract_samples(endname):
    dataset_path = f"/app/data/sampleagreements/{endname}/{endname}"  # Updated path for Render
    docx_files = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isfile(item_path) and item.lower().endswith('.docx'):
            docx_files.append(item_path)
    return docx_files

# Define sample_agreements
sample_agreements = []
for agreement_type in ["rent", "nda", "employment", "franchise", "contractor"]:
    sample_files = extract_samples(agreement_type)
    for file in sample_files:
        doc = Document(file)
        if doc.paragraphs:
            sample_agreements.append(doc.paragraphs[0].text)
sample_agreements = "\n".join(sample_agreements)

# Existing functions
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

@retry.Retry(predicate=is_retriable)
def generate_embeddings(cl, etype):
    if etype:
        embedding_task = "retrieval_document"
    else:
        embedding_task = "retrieval_query"
    embed = client.models.embed_content(
        model="models/text-embedding-004",
        contents=cl,
        config=types.EmbedContentConfig(
            task_type=embedding_task
        )
    )
    return [e.values for e in embed.embeddings]

def pos_neg(response: str):
    prompt = f"""
    Classify the sentiment of the following sentence. 
    Reply with ONLY '1' if the sentence is positive and ONLY '0' if the sentence is negative.

    Sentence = {response}
    """
    response_heat = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=model_config
    )
    return bool(int(response_heat.text))

def strip_type(agr: str):
    agreement_types = ["rent", "nda", "contractor", "employment", "franchise"]
    prompt = f"""Return the type of agreement that the user is referring to in his input {agr}. Respond in one word, all lowecase. You're responses can only be 
    from the set {agreement_types}. Do not use any punctuation. Just respond with the single word."""
    full_prompt = f"""
    Prompt: {prompt}
    
    Possible responses: {agreement_types}

    Sentence: {agr}
    
    Respond in one word, only with type. all lowercase.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=full_prompt,
        config=model_config
    )
    return response.text

def perform_analysis(atype, impt):
    prompt = f""" You are a legal assistant specialising in determining if the input parameters by the user, defined in {impt} are enough parameters to format an agreement of type
    {atype}. 
    
    Please evaluate if the provided information seems to cover all the generally important aspects for a '{atype}' agreement.

    Make sure to evaluate the quality of the input too, if the input seems vague, do consider it as a invalid/bad input.

    Your evaluation must be extremely strict and precise. Any vagueness/lack of information must be considered a severe defect

    Respond with ONLY these messages, no others.
    - "Yes. All essential information seems to be present." if the input appears comprehensive.
    - "No, The following essential information seems to be missing or unclear: [list of missing/unclear aspects]" if key details appear to be absent. Be specific about what's lacking (e.g., "names of all parties", "duration of the agreement", "specific details about the confidential information").
    - "No, The provided information is too vague or insufficient." if the input is very brief or lacks substantial details.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=model_config
    )
    print(response.text)
    global req
    if pos_neg(response.text):
        req = True
    else:
        req = False
        get_user_input()

def obtain_information_holes():
    global important_info, extra_info, final_type
    total_info = important_info + extra_info

    prompt = f"""
        The total information given by the user as an input to generate the agreement of type {final_type} are given in {total_info}

        Identify any missing or unclear information needed to generate a {final_type} agreement based on the provided user input: {total_info}. 
        As a comprehensive legal assistant, pinpoint specific details that require clarification or are absent from the input.

        Generate your final prompt in a way such that if it is passed into a google search, it gives back the required information.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=model_config,
    )
    return response.text

def get_data(holes: str):
    global final_type
    search_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    
    prompt = f"""
    As a LLM, you have identified a few information deficiencies, outlined in {holes} required to generate a LAW agreement of type {final_type}

    You are supposed to retrieve the relevant information using google search. Make sure to keep it concise and accurate.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=search_config,
    )
    return response.text

# Initialize ChromaDB and load clauses
clientdb = chromadb.Client()
rent = clientdb.get_or_create_collection(name="rent_agreements")
nda = clientdb.get_or_create_collection(name="nda_agreements")
employment = clientdb.get_or_create_collection(name="employ_agreements")
franchise = clientdb.get_or_create_collection(name="franch_agreements")
contractor = clientdb.get_or_create_collection(name="contract_agreements")
all_dbs = [rent, nda, employment, franchise, contractor]

rent_clauses = read_docx("rent")
nda_clauses = read_docx("nda")
employ_clauses = read_docx("employment")
franch_clauses = read_docx("franchise")
contract_clauses = read_docx("contractor")
all_clauses = [rent_clauses, nda_clauses, employ_clauses, franch_clauses, contract_clauses]

for j, dataset in enumerate(all_clauses):
    embeds = []
    ids = []
    documents = []
    for i, clause in enumerate(dataset):
        vector = generate_embeddings(clause, True)
        time.sleep(0.4)
        embeds.append(vector[0])
        ids.append(f"clause-{j}-{i}")
        documents.append(clause)
    all_dbs[j].add(embeddings=embeds, ids=ids, documents=documents)

@app.route('/generate-agreement', methods=['POST'])
def generate_agreement():
    try:
        data = request.json
        agreement_type = data.get('agreement_type')
        important_info = data.get('important_info')
        extra_info = data.get('extra_info')

        global final_type, important_info, extra_info, req
        final_type = agreement_type
        important_info = important_info
        extra_info = extra_info
        req = False

        perform_analysis(agreement_type, important_info)
        if req:
            info_holes = obtain_information_holes()
            obtained_info = get_data(info_holes)
            dbname = final_type + "_agreements"
            querydb = clientdb.get_collection(name=dbname)
            query_embed = generate_embeddings(extra_info, False)
            results = querydb.query(query_embeddings=query_embed, n_results=40)
            relevant_documents = results['documents'][0]

            combined_prompt = f"""
                {prompt}

                Sample Agreements:
                {sample_agreements}

                Relevant Clauses:
                {relevant_documents}
            """
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=combined_prompt,
                config=model_config
            )
            return jsonify({"response": response.text})
        else:
            return jsonify({"response": "Please provide more information."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def serve_ui():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
