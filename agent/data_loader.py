# data_loader.py
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def extract_metadata(text_sample: str) -> Dict[str, Any]:
    prompt = PromptTemplate.from_template(
        """
        Analyze the following academic text excerpt (usually the first page) and extract the metadata.
        Return ONLY a valid JSON object with the following exact keys: 
        - 'title' (string)
        - 'authors' (list of strings)
        - 'year' (integer, or null if absolutely not found)
        - 'summary' (a concise 1-sentence summary of the paper)
        - 'page' (integer of the page, or null if it's unavaliable)
        - 'urls' (list of strings for all valid links in the file, keep blank if there is no URLs)
        
        Do not include markdown blocks like ```json. Just the raw JSON.
        
        Text: 
        {text}
        """
    )
    try:
        chain = prompt | llm
        response = chain.invoke({"text": text_sample[:3000]})
        content = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"--- [Metadata] LLM Extraction failed: {e} ---")
        return {"title": "Unknown Title", "authors": [], "year": None, "summary": ""}

def load_and_chunk_pdf(file_path: Path | str, chunk_strategy: str = "semantic") -> List[Dict]:
    file_path = str(file_path)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    if not docs:
        return []

    front_matter = "\n".join([d.page_content for d in docs[:3]])
    print(f"--- [Loader] Calling LLM to extract metadata for {os.path.basename(file_path)} ---")
    llm_metadata = extract_metadata(front_matter)
    
    if not llm_metadata.get("title") or llm_metadata.get("title") == "Unknown Title":
        llm_metadata["title"] = os.path.basename(file_path)

    for doc in docs:
        doc.metadata.update({
            "title": llm_metadata.get("title"),
            "authors": ", ".join(llm_metadata.get("authors", [])),
            "year": llm_metadata.get("year"),
            "summary": llm_metadata.get("summary"),
            "page": llm_metadata.get("page", -1),
            "urls": llm_metadata.get("urls", []),
            "source_type": "local_pdf"
        })

    if chunk_strategy == "semantic":
        text_splitter = SemanticChunker(
            embeddings, 
            breakpoint_threshold_type="percentile" 
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunked_docs = text_splitter.split_documents(docs)
    
    processed_chunks = []
    for i, chunk in enumerate(chunked_docs):
        
        processed_chunks.append({
            "text": chunk.page_content,
            "metadata": chunk.metadata,
            "chunk_index": i
        })
        
    print(f"Processed {os.path.basename(file_path)}: {len(processed_chunks)} chunks (Strategy: {chunk_strategy})")
    return processed_chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Batch embedding wrapper"""
    return embeddings.embed_documents(texts)