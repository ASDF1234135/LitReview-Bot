import arxiv
import requests
import fitz
from typing import List, Dict, Any
import io

def search_arxiv_metadata(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    print(f"--- [Tool] Broad Search ArXiv for: {query} ---")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    try:
        for r in client.results(search):
            results.append({
                "id": r.entry_id.split('/')[-1],
                "title": r.title,
                "summary": r.summary.replace("\n", " "),
                "url": r.pdf_url,
                "published": r.published.strftime("%Y-%m-%d"),
                "authors": [a.name for a in r.authors]
            })
    except Exception as e:
        print(f"ArXiv Search Error: {e}")
        
    return results

def download_and_parse_pdf(pdf_url: str) -> str:
    print(f"--- [Tool] Downloading Full Text: {pdf_url} ---")
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        
        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text = ""
            for page in doc: 
                text += page.get_text()
                
        return text
    except Exception as e:
        print(f"PDF Download Error: {e}")
        return "[Error: Failed to download or parse PDF content]"