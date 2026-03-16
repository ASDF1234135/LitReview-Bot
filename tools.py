# tools.py
import os
import arxiv
import requests
import fitz  # PyMuPDF
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from vector_db import QdrantStorage
from data_loader import get_embeddings

db = QdrantStorage()

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

@tool
def search_arxiv_external(query: str, max_results: int = 10) -> str:
    """
    Broadly search for research papers on ArXiv. 
    Returns ONLY abstracts and IDs. Use 'read_arxiv_paper' to deeply analyze specific IDs found here.
    """
    print(f"--- [Tool] ArXiv Broad Search: {query} ---")
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    
    output = "--- ARXIV SEARCH RESULTS (ABSTRACTS ONLY) ---\n"
    
    for r in client.results(search):
        authors = ", ".join([author.name for author in r.authors][:3])
        year = r.published.year if r.published else "Unknown"
        
        paper_text = (
            f"Title: {r.title}\n"
            f"ArXiv ID: {r.entry_id.split('/')[-1]}\n" # 只取 ID 數字部分
            f"Year: {year} | Authors: {authors}\n"
            f"Summary: {r.summary}\n"
        )
        output += paper_text + "-"*40 + "\n"
        
    return output

@tool
async def read_arxiv_paper(arxiv_id: str, extraction_points: list[str]) -> str:
    """
    Spawns a sub-agent to completely download, read, and extract highly specific information from a single ArXiv paper.
    Use this AFTER finding a relevant paper ID using 'search_arxiv_external'.
    
    Args:
        arxiv_id: The exact ArXiv ID of the paper (e.g., '1801.01234').
        extraction_points: A list of specific, detailed questions or topics to extract. 
                           Break down complex requests into distinct points.
                           (e.g., ["What is the specific experimental design?", "List all hyperparameters used", "What are the limitations?"])
    """
    print(f"--- [Sub-Agent Tool] Deep reading paper ID: {arxiv_id} ---")
    print(f"--- [Sub-Agent Tool] Focus points: {len(extraction_points)} items ---")
    
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        try:
            paper = next(client.results(search))
        except StopIteration:
            return f"Error: Could not find paper with ID {arxiv_id} on ArXiv."
            
        pdf_url = paper.pdf_url
        print(f"--- [Sub-Agent Tool] Downloading PDF from {pdf_url} ---")
        
        pdf_response = requests.get(pdf_url)
        temp_pdf_path = f"temp_{arxiv_id.replace('/', '_')}.pdf"
        
        with open(temp_pdf_path, 'wb') as f:
            f.write(pdf_response.content)
            
        doc = fitz.open(temp_pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            
        formatted_points = "\n".join([f"{i+1}. {pt}" for i, pt in enumerate(extraction_points)])
            
        prompt = PromptTemplate.from_template(
            "You are a meticulous senior academic reviewer.\n"
            "Read the following academic paper text and perform a targeted extraction.\n\n"
            "You MUST address EACH of the following extraction points individually:\n"
            "<extraction_points>\n"
            "{points}\n"
            "</extraction_points>\n\n"
            "Rules:\n"
            "1. Create a clear Markdown heading for each point (e.g., '### 1. Experimental Design').\n"
            "2. Extract highly specific details, especially numbers, algorithms, datasets, and parameters.\n"
            "3. Do NOT hallucinate. If a specific point is not mentioned in the text, write: 'Information not explicitly stated in the paper.'\n\n"
            "Paper Text (Partial/Full):\n"
            "--------------------------\n"
            "{text}\n"
            "--------------------------\n"
            "Provide your highly structured report below:"
        )
        
        chain = prompt | sub_agent_llm
        
        result = await chain.ainvoke({
            "points": formatted_points, 
            "text": full_text[:80000]
        }) 
        
        return f"--- Deep Extraction Report for Paper {arxiv_id} ---\n{result.content}"
        
    except Exception as e:
        return f"Failed to deeply read paper {arxiv_id}. Error: {str(e)}"
    
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