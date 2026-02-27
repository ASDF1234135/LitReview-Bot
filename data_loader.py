# data_loader.py
import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 初始化 Embedding 模型 (用於 Semantic Chunking)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# 初始化 LLM (用於 Metadata 萃取)
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

def extract_metadata(text_sample: str) -> Dict[str, Any]:
    """使用 LLM 從文本前 2000 字萃取 Metadata"""
    prompt = PromptTemplate.from_template(
        """
        Analyze the following academic text excerpt and extract metadata.
        Return ONLY a JSON object with keys: 'title', 'authors' (list of strings), 'year' (int), 'summary' (1 sentence).
        If unknown, use null.
        
        Text: {text}
        """
    )
    try:
        chain = prompt | llm
        response = chain.invoke({"text": text_sample[:2000]})
        # 簡單清洗 json markdown
        import json
        content = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"Metadata extraction failed: {e}")
        return {"title": "Unknown", "authors": [], "year": None, "summary": ""}

def load_and_chunk_pdf(file_path: Path, chunk_strategy: str = "semantic") -> List[Dict]:
    """
    讀取 PDF，萃取 Metadata，並進行語意切片
    """
    loader = PyMuPDFLoader(str(file_path))
    docs = loader.load()
    full_text = "\n".join([d.page_content for d in docs])
    
    # 1. 萃取 Metadata
    metadata = extract_metadata(full_text)
    metadata["source"] = file_path.name
    
    # 2. 語意切片 (Semantic Chunking)
    if chunk_strategy == "semantic":
        # breakpoint_threshold_type="percentile" 表示當語意差異大於某個百分比時切斷
        text_splitter = SemanticChunker(
            embeddings, 
            breakpoint_threshold_type="percentile" 
        )
    else:
        # Fallback to fixed size
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = text_splitter.split_text(full_text)
    
    # 3. 組合結果
    processed_chunks = []
    for i, chunk_text in enumerate(chunks):
        processed_chunks.append({
            "text": chunk_text,
            "metadata": metadata, # 每一塊都帶有全域 Metadata
            "chunk_index": i
        })
        
    print(f"Processed {file_path.name}: {len(processed_chunks)} chunks (Strategy: {chunk_strategy})")
    return processed_chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Batch embedding wrapper"""
    return embeddings.embed_documents(texts)