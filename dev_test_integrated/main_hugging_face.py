import os
import json
import uuid
import re
from typing import Dict, List, Tuple
from enum import Enum
import asyncio # Import asyncio for async operations if needed in the future, though not strictly for this fix

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import pytesseract
from pdf2image import convert_from_path

# Import your existing invoice processing functions from Langchain and Mistral
# Ensure these libraries are installed:
# pip install fastapi uvicorn python-multipart python-dotenv "mistralai>=0.2.0" "langchain-community" "langchain-huggingface" "pytesseract" "pdf2image"
# Also ensure you have Poppler and Tesseract OCR installed on your system.
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from mistralai import Mistral

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_KEY")
APP_API_KEY = os.getenv("API_KEY") # This is your app's main API key for access
MODEL_NAME = "mistral-small-latest"

# Ensure API keys are loaded
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_KEY environment variable not set. Please check your .env file.")
if not APP_API_KEY:
    raise ValueError("API_KEY environment variable not set. Please check your .env file.")

client = Mistral(api_key=MISTRAL_API_KEY)
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

app = FastAPI(
    title="Invoice Processing API",
    description="API for extracting structured data from PDF invoices."
)

# --- API Key Authentication ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Depends(api_key_header)):
    """
    Dependency to validate the API Key provided in the X-API-Key header.
    """
    if api_key == APP_API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "APIKey"},
    )

# --- In-memory Storage for Invoice Processing Status and Results ---
# In a real-world scenario, you would use a database (e.g., Redis, PostgreSQL)
# for persistent storage and better scalability.
class InvoiceStatus(str, Enum):
    RECEIVED = "received"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Dictionary to store the state of each invoice processing task
# Key: invoice_id (str), Value: Dict (status, filled_json, flag_array, error_message, etc.)
invoice_processing_tasks: Dict[str, Dict] = {}

# --- Pydantic Models for API Endpoints ---
class UploadInvoiceResponse(BaseModel):
    invoice_id: str
    status: InvoiceStatus
    message: str

class InvoiceProcessingStatusResponse(BaseModel):
    invoice_id: str
    status: InvoiceStatus
    message: str
    filled_json: Dict[str, str] | None = None
    flag_array: List[int] | None = None
    error_detail: str | None = None

# --- Your original invoice processing functions ---
# These functions are kept as is, but process_invoice_document is renamed
# and adapted to update the in-memory dictionary.

def strip_markdown_code_block(text: str) -> str:
    """Removes markdown code block fences from a string."""
    return re.sub(r"^```.*?$|```", "", text, flags=re.MULTILINE).strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text directly from a PDF using PyPDFLoader."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return "\n\n".join([p.page_content for p in pages])
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def is_text_enough(text: str, min_length=100) -> bool:
    """Checks if the extracted text meets a minimum length."""
    return len(text.strip()) >= min_length

def ocr_pdf_to_text(pdf_path: str) -> str:
    """Performs OCR on a PDF using pdf2image and pytesseract."""
    try:
        # On Windows, you might need to specify the path to tesseract.exe
        # Example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        images = convert_from_path(pdf_path)
        ocr_texts = [pytesseract.image_to_string(img) for img in images]
        return "\n\n".join(ocr_texts)
    except Exception as e:
        print(f"Error performing OCR on PDF {pdf_path}: {e}")
        return ""

def split_into_chunks(text: str):
    """Splits a large text into smaller, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " "]
    )
    return splitter.create_documents([text])

def create_retriever(chunks):
    """Creates a Chroma vectorstore and a retriever from text chunks."""
    # Using a unique collection name for each run to avoid conflicts
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        collection_name=str(uuid.uuid4()),
        persist_directory=None # In-memory
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def retrieve_relevant_context(retriever) -> str:
    """Retrieves the most relevant context from the document using the retriever."""
    results = retriever.invoke("Extract invoice data")
    return "\n\n".join(doc.page_content for doc in results)

def extract_all_fields_at_once(document_text: str, json_template: Dict[str, str]) -> str:
    """Calls the Mistral LLM to extract data based on a prompt and JSON template."""
    prompt = f"""
You are a highly accurate structured data extraction assistant. Your job is to extract structured fields from invoices.

Your task is to read the DOCUMENT provided below and fill in the JSON template with the extracted values. Return ONLY valid JSON matching the template.

Strict rules:
- If a field is not found in the document, leave its value as an empty string.
- Do NOT include any extra text or commentary.
- Do NOT change the field names or JSON structure.
- Return ONLY the JSON object as the answer, with correct JSON syntax.

DOCUMENT:
{document_text}

JSON TEMPLATE:
{json.dumps(json_template, indent=2)}

YOUR RESPONSE:
"""
    response = client.chat.complete(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content
    return response

def run_extraction_pipeline(raw_text: str, json_template: Dict[str, str]) -> Tuple[Dict[str, str], List[int]]:
    """Runs the LLM-based extraction pipeline (chunking, retrieval, LLM call)."""
    chunks = split_into_chunks(raw_text)
    retriever = create_retriever(chunks)
    context = retrieve_relevant_context(retriever)
    raw_llm_response = extract_all_fields_at_once(context, json_template)
    cleaned_response = strip_markdown_code_block(raw_llm_response)

    try:
        filled_json = json.loads(cleaned_response)
    except json.JSONDecodeError:
        filled_json = {} # Return empty on JSON decode error
        print(f"JSONDecodeError: Could not parse LLM response: {cleaned_response}")
        # Optionally log the raw response for debugging
        # with open("extraction_raw_output.txt", "w") as f:
        #     f.write(raw_llm_response)
        return filled_json, []

    # Ensure all keys from template are in filled_json, even if empty, for consistent flagging
    final_filled_json = {k: filled_json.get(k, "") for k in json_template.keys()}
    flag_array = [1 if final_filled_json.get(k, "") else 0 for k in json_template.keys()]
    return final_filled_json, flag_array

# This function will now be run as a background task by FastAPI
def process_invoice_document_background(
    invoice_id: str,
    pdf_path: str,
    json_template: Dict[str, str],
    missing_threshold: float = 0.3
):
    """
    The main invoice processing logic, executed in a background task.
    Updates the global invoice_processing_tasks dictionary with status and results.
    """
    try:
        # Set status to processing
        invoice_processing_tasks[invoice_id]["status"] = InvoiceStatus.PROCESSING
        
        # Initial text extraction attempt
        method_used = "text"
        text = extract_text_from_pdf(pdf_path)

        # Fallback to OCR if direct text is insufficient
        if not is_text_enough(text):
            text = ocr_pdf_to_text(pdf_path)
            method_used = "ocr"

        # Run the extraction pipeline
        filled_json, flag_array = run_extraction_pipeline(text, json_template)

        # Calculate missing rate
        if flag_array:
            missing_rate = flag_array.count(0) / len(flag_array)
        else:
            missing_rate = 1.0 # All fields considered missing if no flags

        # If text method was used and too many fields are missing, retry with OCR
        if missing_rate > missing_threshold and method_used == "text":
            print(f"Retrying invoice {invoice_id} with OCR due to high missing rate ({missing_rate*100:.2f}%) from text extraction.")
            text = ocr_pdf_to_text(pdf_path)
            filled_json, flag_array = run_extraction_pipeline(text, json_template)
            method_used = "ocr"
        
        # Store results and set status to completed
        invoice_processing_tasks[invoice_id].update({
            "status": InvoiceStatus.COMPLETED,
            "filled_json": filled_json,
            "flag_array": flag_array,
            "method_used": method_used # Keep track of which method was finally used
        })

    except Exception as e:
        # Catch any error during processing and update status to failed
        print(f"Error processing invoice {invoice_id}: {e}")
        invoice_processing_tasks[invoice_id].update({
            "status": InvoiceStatus.FAILED,
            "error_detail": str(e)
        })
    finally:
        # Always clean up the temporary PDF file
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except OSError as e:
                print(f"Error removing temporary PDF file {pdf_path}: {e}")

# --- FastAPI Endpoints ---

# ONLY the upload_invoice function needs this change

@app.post("/upload-invoice", response_model=UploadInvoiceResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_invoice(
    background_tasks: BackgroundTasks, # This has no default, so it comes first
    pdf_file: UploadFile = File(..., description="PDF invoice file"),
    json_template_file: UploadFile = File(..., description="JSON template file"),
    api_key: str = Depends(get_api_key) # All arguments with defaults come after
):
    """
    API 1: Uploads an invoice PDF and a JSON template for processing.
    The actual processing is done in a background task.
    Returns a unique invoice ID and status.
    """
    invoice_id = str(uuid.uuid4())
    temp_dir = "temp_invoices"
    temp_pdf_path = os.path.join(temp_dir, f"{invoice_id}.pdf")
    
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Save the PDF file temporarily
        with open(temp_pdf_path, "wb") as buffer:
            buffer.write(await pdf_file.read())

        # Read and parse the JSON template
        json_template_content = await json_template_file.read()
        try:
            json_template = json.loads(json_template_content)
            if not isinstance(json_template, dict):
                raise ValueError("JSON template must be a dictionary (JSON object).")
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON template provided. Please ensure it's valid JSON."
            )
        except ValueError as ve:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON template format: {ve}"
            )

        # Initialize the task status in the global dictionary
        invoice_processing_tasks[invoice_id] = {
            "status": InvoiceStatus.RECEIVED,
            "filled_json": None,
            "flag_array": None,
            "error_detail": None,
            "method_used": None
        }

        # Add the invoice processing to background tasks
        background_tasks.add_task(
            process_invoice_document_background,
            invoice_id,
            temp_pdf_path,
            json_template
        )

        return UploadInvoiceResponse(
            invoice_id=invoice_id,
            status=InvoiceStatus.RECEIVED,
            message="Invoice received and processing initiated."
        )

    except HTTPException as e:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        if invoice_id in invoice_processing_tasks:
            del invoice_processing_tasks[invoice_id]
        raise e
    except Exception as e:
        print(f"Error during file upload or initialization for invoice {invoice_id}: {e}")
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        if invoice_id in invoice_processing_tasks:
            del invoice_processing_tasks[invoice_id]
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to receive or initialize invoice processing: {e}"
        )
    
@app.get("/invoice-status/{invoice_id}", response_model=InvoiceProcessingStatusResponse)
async def get_invoice_status(
    invoice_id: str,
    api_key: str = Depends(get_api_key) # Protect this endpoint with API key
):
    """
    API 2: Checks the status of an invoice processing task.
    If completed, returns the filled JSON and flag array.
    """
    task_info = invoice_processing_tasks.get(invoice_id)

    if not task_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invoice with ID '{invoice_id}' not found or processing has not started."
        )

    current_status = task_info.get("status")

    if current_status == InvoiceStatus.COMPLETED:
        return InvoiceProcessingStatusResponse(
            invoice_id=invoice_id,
            status=InvoiceStatus.COMPLETED,
            message="Processing complete.",
            filled_json=task_info.get("filled_json"),
            flag_array=task_info.get("flag_array")
        )
    elif current_status == InvoiceStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invoice processing failed: {task_info.get('error_detail')}",
            headers={"X-Invoice-Status": InvoiceStatus.FAILED.value} # Custom header for programmatic checking
        )
    else:
        # For RECEIVED or PROCESSING statuses
        return InvoiceProcessingStatusResponse(
            invoice_id=invoice_id,
            status=current_status,
            message=f"Invoice processing is {current_status.value}."
        )

# Example Usage (for local testing, can be removed in production)
@app.get("/")
async def root():
    return {"message": "Invoice Processing API is running. Check /docs for API documentation."}