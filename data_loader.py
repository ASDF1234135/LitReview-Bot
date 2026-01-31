from google import genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_chunk_pdf(path):
    docs = PDFReader().load_data(path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts):
    all_embeddings = []
    batch_size = 100
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch
            )
            batch_embeddings = [item.values for item in response.embeddings]
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"Error embedding batch {i}-{i+batch_size}: {e}")
            raise e

    return all_embeddings

def chunk_text(text: str) -> list[str]:
    return splitter.split_text(text)