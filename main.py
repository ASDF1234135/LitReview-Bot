# main.py 的最頂部
import sys
import asyncio

# --- 解決 Windows Asyncio 與 Psycopg3 的衝突 ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import uvicorn
import inngest
import inngest.fast_api
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from agent_core import get_thread_history
from fastapi.responses import StreamingResponse
from agent_core import run_research_agent_stream, get_thread_history

# 1. 載入環境變數 (必須在最上面)
load_dotenv()

# 引入我們的新模組
from agent_core import run_research_agent
from vector_db import QdrantStorage
from data_loader import load_and_chunk_pdf, get_embeddings

# --- 配置 ---
inngest_client = inngest.Inngest(app_id="rag_agent_app", is_production=False)

app = FastAPI(title="AI Research Agent API")
db = QdrantStorage()


# 允許跨域 (配合 Streamlit 開發)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    thread_id: str = "thread_1"  # 用於記憶對話歷史

class IngestRequest(BaseModel):
    file_path: str
    user_id: str

# --- Inngest Functions (背景任務) ---

@inngest_client.create_function(
    fn_id="RAG: Ingest User PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    concurrency=[inngest.Concurrency(limit=2)] # 限制同時處理的 PDF 數量，避免 OOM
)
# 【唯一修正這裡】：只留下一個 ctx: inngest.Context 參數
async def rag_ingest_pdf(ctx: inngest.Context):
    """
    背景任務：當使用者上傳 PDF 時，自動進行語意切片並存入 Qdrant
    """
    # 1. 從 ctx 提取 event data
    data = ctx.event.data
    pdf_path = Path(data["pdf_path"])
    user_id = data["user_id"]

    # 2. 從 ctx 提取 step 來執行任務 (加上 ctx. 前綴)
    chunks = await ctx.step.run("parse_and_chunk", lambda: load_and_chunk_pdf(
        file_path=pdf_path, 
        chunk_strategy="semantic"
    ))

    if not chunks:
        return {"status": "error", "message": "No text extracted from PDF"}

    # Step 2: 準備向量與 Metadata
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Step 3: 計算向量 (Batch Embedding)
    vectors = await ctx.step.run("generate_embeddings", lambda: get_embeddings(texts))

    # Step 4: 存入 Qdrant (標記為 Private)
    def save_to_db():
        db.upsert(
            texts=texts,
            metadatas=metadatas,
            vectors=vectors,
            user_id=user_id,
            access="private"
        )
        return "Success"

    result = await ctx.step.run("upsert_to_qdrant", save_to_db)

    return {
        "status": "completed",
        "chunks_processed": len(chunks),
        "source": pdf_path.name
    }

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "AI Research Agent is running"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    使用 StreamingResponse 將 Agent 的思考過程以 SSE 格式推播給前端
    """
    try:
        # 將 Generator 丟給 StreamingResponse，並設定 media_type
        return StreamingResponse(
            run_research_agent_stream(request.message, request.user_id, request.thread_id),
            media_type="application/x-ndjson"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{thread_id}")
async def get_history_endpoint(thread_id: str):
    """前端用來拉取歷史對話的 API"""
    try:
        history = await get_thread_history(thread_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 讓 Streamlit 可以觸發 PDF 入庫的簡單接口
@app.post("/api/trigger-ingest")
async def trigger_ingest(request: IngestRequest):
    await inngest_client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": request.file_path,
                "user_id": request.user_id
            }
        )
    )
    return {"status": "Ingestion event dispatched"}

@app.get("/api/files/{user_id}")
async def get_files_endpoint(user_id: str):
    try:
        files = db.get_user_files(user_id)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# [NEW] 刪除檔案 API
@app.delete("/api/files/{user_id}/{filename}")
async def delete_file_endpoint(user_id: str, filename: str):
    try:
        success = db.delete_user_file(user_id, filename)
        if success:
            return {"status": "success", "message": f"Deleted {filename}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete from DB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 註冊 Inngest Handler
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)