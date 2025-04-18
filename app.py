import os
import time
import chromadb
from docx import Document
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.api_core import retry
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GOOGLE_API_KEY = "AIzaSyAQeDHegBbQZ_0-oGS_KxvBPrGvU9Nfcns"  # Consider using environment variables
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')
model_config = genai.types.GenerationConfig(temperature=0.75, top_p=0.9)

# File Path Setup
BASE_DIR = Path(__file__).parent
CLAUSES_DIR = BASE_DIR / "Clauses"
SAMPLES_DIR = BASE_DIR / "sampleagreements"

# ChromaDB Setup
clientdb = chromadb.PersistentClient(path=str(BASE_DIR / "chromadb_data"))
dbs = {
    "rent": clientdb.get_or_create_collection(name="rent_agreements"),
    "nda": clientdb.get_or_create_collection(name="nda_agreements"),
    "employment": clientdb.get_or_create_collection(name="employ_agreements"),
    "franchise": clientdb.get_or_create_collection(name="franch_agreements"),
    "contractor": clientdb.get_or_create_collection(name="contract_agreements")
}

class AgreementRequest(BaseModel):
    agreement_type: str
    important_info: str
    extra_info: str

# Helper Functions
def read_docx(file_path):
    try:
        return [p.text for p in Document(file_path).paragraphs if p.text.strip()]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading {file_path}: {str(e)}")

def get_sample_paths(agreement_type):
    type_dir = SAMPLES_DIR / agreement_type / agreement_type
    if not type_dir.exists():
        raise HTTPException(status_code=404, detail=f"Sample directory not found: {type_dir}")
    return [str(f) for f in type_dir.glob("*.docx")]

@retry.Retry(predicate=lambda e: isinstance(e, genai.APIError) and e.code in {429, 503})
def generate_embeddings(content, is_doc=True):
    try:
        embed = genai.embed_content(
            model='models/text-embedding-004',
            content=content,
            task_type="retrieval_document" if is_doc else "retrieval_query"
        )
        return [e.values for e in embed.embeddings][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

def initialize_db():
    for agreement_type, db in dbs.items():
        clause_file = CLAUSES_DIR / f"{agreement_type}.docx"
        if not clause_file.exists():
            raise HTTPException(status_code=404, detail=f"Clause file not found: {clause_file}")
        
        clauses = read_docx(clause_file)
        try:
            db.add(
                embeddings=[generate_embeddings(c) for c in clauses],
                ids=[f"{agreement_type}-{i}" for i in range(len(clauses))],
                documents=clauses
            )
            time.sleep(0.4)  # Rate limiting
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB initialization failed: {str(e)}")

# Initialize database on startup
initialize_db()

# API Endpoint
@app.post("/generate-agreement")
async def generate_agreement(request: AgreementRequest):
    try:
        # Standardize agreement type
        agreement_type = model.generate_content(
            f"Return agreement type (rent/nda/contractor/employment/franchise). Input: {request.agreement_type}",
            generation_config=model_config
        ).text.strip().lower()

        # Validate input sufficiency
        analysis = model.generate_content(
            f"Evaluate if '{request.important_info}' is sufficient for '{agreement_type}' agreement. "
            "Respond 'Yes. All essential information seems to be present.' or list missing info.",
            generation_config=model_config
        ).text
        
        if "Yes" not in analysis:
            return {"response": "Please provide more information.", "error": analysis}

        # Retrieve relevant clauses
        db = dbs.get(agreement_type)
        if not db:
            raise HTTPException(status_code=400, detail="Invalid agreement type")
            
        results = db.query(
            query_embeddings=[generate_embeddings(request.extra_info, False)],
            n_results=40
        )
        relevant_clauses = results['documents'][0] if results['documents'] else []

        # Generate agreement
        agreement = model.generate_content(
            f"""Generate {agreement_type} agreement with:
            - Key info: {request.important_info}
            - Extra details: {request.extra_info}
            - Relevant clauses: {relevant_clauses}
            - Samples: {get_sample_paths(agreement_type)}
            """,
            generation_config=model_config
        ).text

        return {"response": agreement, "error": None}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agreement generation failed: {str(e)}")

# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)