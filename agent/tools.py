# tools.py
import os
import requests
import fitz  # PyMuPDF
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from DB.vector_db import QdrantStorage
from agent.data_loader import get_embeddings
from dotenv import load_dotenv

db = QdrantStorage()
load_dotenv()
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")


sub_agent_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

@tool
def search_knowledge_base(query: str, config: RunnableConfig, strategy: str = "hybrid") -> str:
    """
    Search documents in the local database (PDFs uploaded by user).
    Use this to compare external findings with the user's private projects or notes.
    
    Args:
        query: The search query. Can be a specific question or keywords.
        strategy: 'hybrid' for high-precision search (Dense + BM25), or 'dense' for broad semantic search.
    """
    user_id = config["configurable"].get("user_id")
    
    print(f"--- [Tool] Local Search ({strategy}): {query} (User: {user_id}) ---")
    vec = get_embeddings([query])[0]
    results = db.search(query_vector=vec, query_text=query, top_k=5, user_id=user_id, strategy=strategy)
    
    if not results:
        return "No relevant documents found in local private database."
    
    output = "--- PRIVATE KNOWLEDGE BASE RESULTS ---\n"
    for r in results:
        meta = r['metadata']
        title = meta.get('title', 'Unknown')
        year = meta.get('year', 'N/A')
        source = meta.get('source') or meta.get('file_path', 'Unknown') 
        authors = meta.get('authors', 'Unknown')
        
        urls = meta.get('extracted_urls') or meta.get('urls') or []
        url_str = f" | URLs: {', '.join(urls)}" if urls else ""
        
        doi = meta.get('doi', [])
        doi_str = f" | DOI: {', '.join(doi)}" if doi else ""

        page = meta.get('page', None)
        page_str = f" | Page: {', '.join(doi)}" if page else ""
        
        output += f"[Source: {source} | Title: {title} | Authors: {authors} | Year: {year}{page_str}{url_str}{doi_str}]\n"
        
        text_content = r.get('text') or meta.get('text', '')
        output += f"Content: {text_content}\n\n"
    return output

    
ddg_search = DuckDuckGoSearchRun()

@tool
def search_web_general(query: str) -> str:
    """
    Search the general internet for broad topics, history, tech news, or general concepts.
    Use this when the user asks for historical context, overviews, or topics where ArXiv is too specific.
    """
    print(f"--- [Tool] Web Search: {query} ---")
    try:
        result = ddg_search.invoke(query)
        return result
    except Exception as e:
        return f"Web search failed: {str(e)}"

def reconstruct_abstract(inverted_index: dict) -> str:
    """
    OpenAlex API 不會直接回傳完整摘要字串，而是回傳 Inverted Index (倒排索引)。
    這個函數用來將它還原回人類可讀的完整段落。
    """
    if not inverted_index:
        return "No abstract available."
    
    # 找出最大的索引值來建立陣列
    max_idx = max([idx for positions in inverted_index.values() for idx in positions])
    words = [""] * (max_idx + 1)
    
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
            
    return " ".join(words)


@tool
def search_openalex_external(query: str, max_results: int = 5):
    """
    Broad sweep of OpenAlex to find candidate academic papers.
    Returns abstracts, Open Access PDF links, and OpenAlex IDs.
    """
    print(f"--- [OpenAlex Tool] Fetching papers for: '{query}' ---")
    
    base_url = "https://api.openalex.org/works"
    
    query_params = {
        "search": query,
        "per-page": max_results,
        "sort": "relevance_score:desc"
    }

    if OPENALEX_API_KEY:
        query_params["api_key"] = OPENALEX_API_KEY

    try:
        response = requests.get(base_url, params=query_params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            for work in data.get("results", []):
                title = work.get("title", "Unknown Title")
                
                raw_id = work.get("id", "")
                openalex_id = raw_id.split("/")[-1] if raw_id else "Unknown ID"
                
                oa_url = work.get("open_access", {}).get("oa_url", "No Free PDF available")
                
                abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
                
                results.append(
                    f"Title: {title}\n"
                    f"OpenAlex ID: {openalex_id}\n"
                    f"PDF Link: {oa_url}\n"
                    f"Abstract: {abstract}"
                )
                
            if not results:
                return "No papers found on OpenAlex for this query."
                
            return "\n\n---\n\n".join(results)
            
        elif response.status_code == 429:
             return "System Status: OpenAlex API rate limit exceeded. Please rely on private knowledge."
        else:
             return f"OpenAlex API Error: HTTP {response.status_code}"
             
    except Exception as e:
        return f"Tool execution failed due to network error: {str(e)}. Please pivot to private knowledge."

@tool
def read_openalex_paper(openalex_id: str, extraction_points: list[str]) -> str:
    """
    Spawns a sub-agent to completely download, read, and extract highly specific information from a single OpenAlex paper.
    Use this AFTER finding a relevant OpenAlex ID using 'search_openalex_external'.
    
    Args:
        openalex_id: The exact OpenAlex ID of the paper (e.g., 'W2741809807').
        extraction_points: A list of specific, detailed questions or topics to extract.
    """
    print(f"--- [Sub-Agent Tool] Deep reading OpenAlex ID: {openalex_id} ---")
    
    try:
        work_url = f"https://api.openalex.org/works/{openalex_id}"

        query_params = {}
        if OPENALEX_API_KEY:
            query_params["api_key"] = OPENALEX_API_KEY
            
        meta_res = requests.get(work_url, params=query_params, timeout=10)
        
        if meta_res.status_code != 200:
            return f"Error: Could not retrieve metadata for OpenAlex ID {openalex_id}."
            
        work_data = meta_res.json()
        pdf_url = work_data.get("open_access", {}).get("oa_url")
        
        if not pdf_url:
            return f"Error: The paper {openalex_id} is behind a paywall or has no Open Access PDF available for deep reading."
            
        print(f"--- [Sub-Agent Tool] Downloading PDF from {pdf_url} ---")
        
        pdf_response = requests.get(pdf_url, headers=OPENALEX_HEADERS, timeout=15)
        
        if "application/pdf" not in pdf_response.headers.get('Content-Type', ''):
            print("Warning: Retrieved URL is not a direct PDF. PyMuPDF might fail to parse.")

        temp_pdf_path = f"temp_{openalex_id}.pdf"
        
        with open(temp_pdf_path, 'wb') as f:
            f.write(pdf_response.content)
            
        # 3. 解析 PDF
        doc = fitz.open(temp_pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        # 清理暫存檔
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            
        if not full_text.strip():
             return f"Error: Failed to extract readable text from the downloaded file. It might be a scanned image or protected."
            
        # 4. 呼叫 LLM 進行資訊萃取 (沿用您原本完美的 Prompt)
        formatted_points = "\n".join([f"{i+1}. {pt}" for i, pt in enumerate(extraction_points)])
            
        prompt = PromptTemplate.from_template(
            "You are a meticulous senior academic reviewer.\n"
            "Read the following academic paper text and perform a targeted extraction.\n\n"
            "You MUST address EACH of the following extraction points individually:\n"
            "<extraction_points>\n"
            "{points}\n"
            "</extraction_points>\n\n"
            "Rules:\n"
            "1. Create a clear Markdown heading for each point.\n"
            "2. Extract highly specific details, especially numbers, algorithms, datasets, and parameters.\n"
            "3. Do NOT hallucinate. If not stated, write: 'Information not explicitly stated.'\n\n"
            "Paper Text:\n"
            "--------------------------\n"
            "{text}\n"
            "--------------------------\n"
            "Report:"
        )
        
        chain = prompt | sub_agent_llm
        
        result = chain.invoke({
            "points": formatted_points, 
            "text": full_text[:80000]
        }) 
        
        return f"--- Deep Extraction Report for Paper {openalex_id} ---\n{result.content}"
        
    except Exception as e:
        return f"Failed to deeply read paper {openalex_id}. Error: {str(e)}"