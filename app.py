import os
import time
import chromadb
import numpy as np
import pandas as pd
from docx import Document
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.api_core import retry
import google.generativeai as genai
from google.generativeai import types
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client with embedded API key
GOOGLE_API_KEY = "AIzaSyAQeDHegBbQZ_0-oGS_KxvBPrGvU9Nfcns"
genai.configure(api_key=GOOGLE_API_KEY)
client = genai.GenerativeAI()

# Gemini model configuration
model_config = types.GenerationConfig(
    temperature=0.75,
    top_p=0.9,
)

# ChromaDB client
clientdb = chromadb.PersistentClient(path="./chromadb_data")
rent = clientdb.get_or_create_collection(name="rent_agreements")
nda = clientdb.get_or_create_collection(name="nda_agreements")
employment = clientdb.get_or_create_collection(name="employ_agreements")
franchise = clientdb.get_or_create_collection(name="franch_agreements")
contractor = clientdb.get_or_create_collection(name="contract_agreements")
all_dbs = [rent, nda, employment, franchise, contractor]

# Global variables
req = False
final_type = ""
important_info = ""
extra_info = ""
obtained_info = ""

# Input model for API
class AgreementRequest(BaseModel):
    agreement_type: str
    important_info: str
    extra_info: str

# Retry mechanism for API errors
def is_retriable(e):
    return isinstance(e, genai.APIError) and e.code in {429, 503}

# Read DOCX file
def read_docx(endname):
    path = f"./Clauses/{endname}.docx"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return paragraphs

# Extract full DOCX document
def extract_docx(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    doc = Document(path)
    return doc

# List sample agreement paths
def extract_samples(endname):
    dataset_path = f"./sampleagreements/{endname}/{endname}"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Directory {dataset_path} not found")
    docx_files = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isfile(item_path) and item.lower().endswith('.docx'):
            docx_files.append(item_path)
    return docx_files

# Generate text embeddings with retry
@retry.Retry(predicate=is_retriable)
def generate_embeddings(cl, etype):
    embedding_task = "retrieval_document" if etype else "retrieval_query"
    embed = client.embed_content(
        model="models/text-embedding-004",
        content=cl,
        task_type=embedding_task
    )
    return [e.values for e in embed.embeddings]

# Initialize ChromaDB with clauses
def initialize_chromadb():
    agreement_types = ["rent", "nda", "employment", "franchise", "contractor"]
    all_clauses = [read_docx(atype) for atype in agreement_types]
    for j, dataset in enumerate(all_clauses):
        embeds, ids, documents = [], [], []
        for i, clause in enumerate(dataset):
            vector = generate_embeddings(clause, True)
            time.sleep(0.4)  # Avoid API rate limits
            embeds.append(vector[0])
            ids.append(f"clause-{j}-{i}")
            documents.append(clause)
        all_dbs[j].add(embeddings=embeds, ids=ids, documents=documents)

# Standardize agreement type
def strip_type(agr: str):
    agreement_types = ["rent", "nda", "contractor", "employment", "franchise"]
    prompt = f"""Return the type of agreement in one lowercase word from {agreement_types}. Input: {agr}"""
    response = client.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        generation_config=model_config
    )
    return response.text.strip()

# Sentiment analysis for response
def pos_neg(response: str):
    prompt = f"""Classify sentiment. Reply '1' for positive, '0' for negative. Sentence: {response}"""
    response_heat = client.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        generation_config=model_config
    )
    return bool(int(response_heat.text))

# Analyze input sufficiency
def perform_analysis(atype, impt):
    prompt = f"""As a legal assistant, evaluate if the input '{impt}' is sufficient for a '{atype}' agreement.
    Respond with:
    - "Yes. All essential information seems to be present."
    - "No, The following essential information seems to be missing or unclear: [list]"
    - "No, The provided information is too vague or insufficient."
    """
    response = client.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        generation_config=model_config
    )
    return response.text

# Identify missing information
def obtain_information_holes(important_info, extra_info, final_type):
    total_info = important_info + " " + extra_info
    prompt = f"""Identify missing or unclear information for a {final_type} agreement based on: {total_info}."""
    response = client.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        generation_config=model_config
    )
    return response.text

# Simulate data retrieval
def get_data(holes: str, final_type: str):
    prompt = f"""Simulate retrieving information for {holes} to generate a {final_type} agreement."""
    response = client.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        generation_config=model_config
    )
    return response.text

# Initialize ChromaDB on startup
initialize_chromadb()

# API endpoint to generate agreement
@app.post("/generate-agreement")
async def generate_agreement(request: AgreementRequest):
    global req, final_type, important_info, extra_info, obtained_info
    try:
        # Set global variables
        final_type = strip_type(request.agreement_type)
        important_info = request.important_info
        extra_info = request.extra_info
        req = False

        # Validate input sufficiency
        analysis_result = perform_analysis(final_type, important_info)
        if not pos_neg(analysis_result):
            return {"response": "Please provide more information.", "error": analysis_result}

        # Identify and fill information gaps
        info_holes = obtain_information_holes(important_info, extra_info, final_type)
        obtained_info = get_data(info_holes, final_type)

        # Retrieve relevant clauses
        dbname = final_type + "_agreements"
        querydb = clientdb.get_collection(name=dbname)
        query_embed = generate_embeddings(extra_info, False)
        results = querydb.query(query_embeddings=query_embed, n_results=40)
        relevant_documents = results['documents'][0] if results['documents'] else []

        # Get sample agreements
        sample_agreements = extract_samples(final_type)

        # Construct final prompt
        prompt = f"""You are a helpful AI assistant for law agreement generation.
        User info: {important_info}
        Agreement type: {final_type}
        Relevant clauses: {relevant_documents}
        User instructions: {extra_info}
        Sample agreement paths: {sample_agreements}
        Additional info: {obtained_info}
        Generate a concise, legally sound {final_type} agreement.
        """

        # Generate agreement
        response = client.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            generation_config=model_config
        )

        return {"response": response.text, "error": None}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve frontend
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="static", html=True), name="static")