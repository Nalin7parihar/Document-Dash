from fastapi import APIRouter, UploadFile, File,HTTPException,status
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromy import collection
import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import shutil,os
import tiktoken
router = APIRouter(tags=["upload"],prefix="/upload")



tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text:str)->int:
  return len(tokenizer.encode(text))
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with OCR fallback"""
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                text_pages.append({
                    "page_num": page_num,
                    "text": text
                })
            else:
                # Fallback to OCR
                images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
                if images:
                    ocr_text = pytesseract.image_to_string(images[0])
                    text_pages.append({
                        "page_num": page_num,
                        "text": ocr_text
                    })
    
    return text_pages




def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks with proper overlap"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
    )
    return splitter.split_text(text)
  

@router.post("/", status_code=status.HTTP_200_OK)
async def upload_file(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Extract text from PDF (now returns list of page objects)
    pages = extract_text_from_pdf(temp_path)
    
    all_chunks = []
    all_metadatas = []
    
    # Process each page
    for page_data in pages:
        page_num = page_data["page_num"]
        page_text = page_data["text"]
        
        if page_text and page_text.strip():
            # Split page into chunks
            chunks = chunk_text(page_text)
            
            for chunk_index, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "source": file.filename,
                        "page_number": page_num,
                        "chunk_index": chunk_index,
                        "total_chunks": len(chunks)
                    })
    
    # Add to ChromaDB
    if all_chunks:
        collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=[f"{file.filename}_page{meta['page_number']}_chunk{meta['chunk_index']}" 
                 for meta in all_metadatas]
        )
    
    os.remove(temp_path)
    
    return {
        "filename": file.filename,
        "chunks_added": len(all_chunks),
        "pages_processed": len(pages)
    }
  