# tools.py
from langchain_core.tools import tool
from vector_db import QdrantStorage
from data_loader import get_embeddings
from langchain_core.runnables import RunnableConfig
# 假設您原本有 arxiv_client，若無可使用簡單 wrapper
import arxiv 

db = QdrantStorage()

@tool
def search_knowledge_base(query: str, config: RunnableConfig, strategy: str = "dense") -> str:
    """
    Search documents in the local database (PDFs uploaded by user).
    
    Args:
        query: The search query. Can be a specific question or keywords.
        strategy: 'dense' for broad semantic search, or 'hybrid' for high-precision search.
    """
    # [關鍵] 從底層 Config 偷偷拿出 user_id，不讓 LLM 煩惱這個
    user_id = config["configurable"].get("user_id")
    
    print(f"--- [Tool] Local Search ({strategy}): {query} (User: {user_id}) ---")
    vec = get_embeddings([query])[0]
    results = db.search(query_vector=vec, query_text=query, top_k=5, user_id=user_id, strategy=strategy)
    
    if not results:
        return "No relevant documents found in local database."
    
    output = ""
    for r in results:
        meta = r['metadata']
        output += f"[Title: {meta.get('title', 'Unknown')} | Year: {meta.get('year', 'N/A')}]\n"
        output += f"Content: {r['text']}\n\n"
    return output

@tool
def search_arxiv_external(query: str, config: RunnableConfig, max_results: int = 3) -> str:
    """
    Search for NEW research papers on ArXiv API. 
    Results are automatically saved to the local knowledge base.
    """
    # [關鍵] 從底層 Config 偷偷拿出 user_id
    user_id = config["configurable"].get("user_id")
    
    print(f"--- [Tool] ArXiv Search: {query} (User: {user_id}) ---")
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    
    output = ""
    texts_to_save = []
    metadatas_to_save = []
    
    for r in client.results(search):
        paper_text = f"Title: {r.title}\nID: {r.entry_id}\nSummary: {r.summary[:500]}..."
        output += paper_text + "\n\n"
        
        texts_to_save.append(paper_text)
        metadatas_to_save.append({
            "title": r.title,
            "year": r.published.year if r.published else 2026,
            "authors": [author.name for author in r.authors][:3],
            "source_type": "arxiv"
        })
    
    if texts_to_save:
        print(f"--- [DB] Auto-saving {len(texts_to_save)} ArXiv papers to Qdrant ---")
        vecs = get_embeddings(texts_to_save)
        db.upsert(
            texts=texts_to_save, metadatas=metadatas_to_save, vectors=vecs,
            user_id=user_id, access="public"
        )

    return output