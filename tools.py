import arxiv
from typing import List, Dict, Any

def search_arxiv_papers(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    print(f"--- [Tool] Searching ArXiv for: {query} ---")
    
    client = arxiv.Client()
    
    search = arxiv.Search(
        query = query,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )
    
    results = []
    try:
        for r in client.results(search):
            results.append({
                "title": r.title,
                "summary": r.summary.replace("\n", " "),
                "url": r.pdf_url,
                "published": r.published.strftime("%Y-%m-%d"),
                "authors": [a.name for a in r.authors]
            })
    except Exception as e:
        print(f"ArXiv Search Error: {e}")
        
    return results

def format_arxiv_to_context(papers: List[Dict[str, Any]]) -> List[str]:
    contexts = []
    for p in papers:
        text = (
            f"[Source: {p['url']}]\n"
            f"Title: {p['title']}\n"
            f"Published: {p['published']}\n"
            f"Abstract: {p['summary']}"
        )
        contexts.append(text)
    return contexts