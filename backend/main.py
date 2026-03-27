import sys
import asyncio

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
from agent.agent_core import get_thread_history
from fastapi.responses import StreamingResponse
from agent.agent_core import run_research_agent_stream, get_thread_history
from fastapi.middleware.cors import CORSMiddleware
from agent.agent_core import run_research_agent
from DB.vector_db import QdrantStorage
from agent.data_loader import load_and_chunk_pdf, get_embeddings
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
import bcrypt
from pydantic import BaseModel
from DB.database import get_db, User, ChatThread
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from typing import List
import datetime
import jwt
from fastapi import Security
from fastapi.security import OAuth2PasswordBearer
import time
import psutil

load_dotenv()

IS_CLOUD = os.getenv("RENDER") is not None
inngest_client = inngest.Inngest(app_id="rag_agent_app", is_production=IS_CLOUD)

app = FastAPI(title="AI Research Agent API")
db = QdrantStorage()
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_please_change_in_env")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")


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

class ThreadCreate(BaseModel):
    username: str
    thread_id: str
    title: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

# -- Helpers --
def get_password_hash(password: str) -> str:
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(pwd_bytes, salt)
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode('utf-8')
    hashed_password_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_password_bytes)

async def get_current_admin(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401, detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        
        if username is None or role is None:
            raise credentials_exception
            
        if role != "admin":
            raise HTTPException(status_code=403, detail="Not enough permissions")
            
        return payload
        
    except jwt.PyJWTError:
        raise credentials_exception
    
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or token expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        username: str = payload.get("sub")
        
        if username is None:
            raise credentials_exception
            
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please log in again.")
    except jwt.PyJWTError:
        raise credentials_exception

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Inngest Functions (背景任務) ---

@inngest_client.create_function(
    fn_id="RAG: Ingest User PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    concurrency=[inngest.Concurrency(limit=2)] # 限制同時處理的 PDF 數量，避免 OOM
)
async def rag_ingest_pdf(ctx: inngest.Context):
    # 1. 從 ctx 提取 event data
    data = ctx.event.data
    pdf_path = Path(data["pdf_path"])
    user_id = data["user_id"]

    async def process_entire_pipeline():
        chunks = load_and_chunk_pdf(
            file_path=pdf_path, 
            chunk_strategy="semantic"
        )

        if not chunks:
            return {"status": "error", "message": "No text extracted from PDF"}

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        vectors = get_embeddings(texts) 

        db.upsert(
            texts=texts,
            metadatas=metadatas,
            vectors=vectors,
            user_id=user_id,
            access="private"
        )
        
        try:
            if pdf_path.exists():
                os.remove(pdf_path)
                print(f"--- [Cleanup] Successfully deleted temp file: {pdf_path.name} ---")
        except Exception as e:
            print(f"--- [Cleanup Error] Failed to delete {pdf_path}: {e} ---")

        return {
            "status": "completed",
            "chunks_processed": len(chunks),
            "source": pdf_path.name
        }

    result = await ctx.step.run("process_and_store_pdf", process_entire_pipeline)

    return result

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "AI Research Agent is running"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    if request.user_id != current_user.get("sub"):
        raise HTTPException(status_code=403, detail="Forbidden: User ID mismatch")
    
    try:
        # 將 Generator 丟給 StreamingResponse，並設定 media_type
        return StreamingResponse(
            run_research_agent_stream(request.message, request.user_id, request.thread_id),
            media_type="application/x-ndjson"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{thread_id}")
async def get_history_endpoint(thread_id: str, current_user: dict = Depends(get_current_user)):
    db_thread = db.query(ChatThread).filter(ChatThread.thread_id == thread_id).first()
    if db_thread and db_thread.username != current_user.get("sub"):
        raise HTTPException(status_code=403, detail="Forbidden: You don't own this thread")
    
    try:
        history = await get_thread_history(thread_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trigger-ingest")
async def trigger_ingest(
    files: List[UploadFile] = File(...), 
    user_id: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    if user_id != current_user.get("sub"):
        raise HTTPException(status_code=403, detail="Forbidden")
    
    events_to_dispatch = []
    os.makedirs("temp_uploads", exist_ok=True)
    
    for file in files:
        temp_file_path = f"temp_uploads/{user_id}_{file.filename}"
        with open(temp_file_path, "wb+") as f:
            f.write(await file.read())
            
        events_to_dispatch.append(
            inngest.Event(
                name="rag/ingest_pdf", 
                data={
                    "user_id": user_id,
                    "pdf_path": temp_file_path,
                    "filename": file.filename
                }
            )
        )
        
    await inngest_client.send(events_to_dispatch)
    
    return {"status": "success", "message": f"Successfully dispatched {len(files)} background jobs."}

@app.get("/api/files/{user_id}")
async def get_files_endpoint(user_id: str, current_user: dict = Depends(get_current_user)):
    if user_id != current_user.get("sub"): 
        raise HTTPException(status_code=403, detail="Forbidden")
    
    try:
        files = db.get_user_files(user_id)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/files/{user_id}/{filename}")
async def delete_file_endpoint(user_id: str, filename: str, current_user: dict = Depends(get_current_user)):
    if user_id != current_user.get("sub"): 
        raise HTTPException(status_code=403, detail="Forbidden")
    
    try:
        success = db.delete_user_file(user_id, filename)
        if success:
            return {"status": "success", "message": f"Deleted {filename}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete from DB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()
    
    if db_user:
        raise HTTPException(
            status_code=400, 
            detail="The account or email has been registered."
        )

    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=get_password_hash(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": "Register Successed", "username": new_user.username}

@app.get("/api/admin/health", dependencies=[Depends(get_current_admin)])
async def get_system_health(db_session: Session = Depends(get_db)):
    start_time = time.time()
    
    cpu_usage = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    ram_usage = ram.percent
    
    ram_note = "Host Memory (Docker Context)" if IS_CLOUD else "Local Memory"

    disk = psutil.disk_usage('/')
    disk_usage = disk.percent
    disk_total_gb = round(disk.total / (1024**3), 2)
    disk_free_gb = round(disk.free / (1024**3), 2)

    db_start = time.time()
    try:
        total_users = db_session.query(User).count()
        total_threads = db_session.query(ChatThread).count()
        db_status = "Online"
    except Exception as e:
        total_users, total_threads = 0, 0
        db_status = f"Offline"
        print(f"DB Error: {e}")
    db_latency = int((time.time() - db_start) * 1000)
    
    qdrant_start = time.time()
    try:
        collection_info = db.client.get_collection(db.collection)
        vector_count = collection_info.points_count
        qdrant_status = "Online"
    except Exception as e:
        vector_count = 0
        qdrant_status = "Offline"
        print(f"Qdrant Error: {e}")
    qdrant_latency = int((time.time() - qdrant_start) * 1000)
    
    api_latency = int((time.time() - start_time) * 1000)

    current_env = "Production (Render)" if IS_CLOUD else "Local Development"
    inngest_dashboard_url = "https://app.inngest.com/" if IS_CLOUD else "http://localhost:8288"

    return {
        "status": "healthy" if db_status == "Online" and qdrant_status == "Online" else "degraded",
        "environment": current_env,
        "api_latency_ms": api_latency,
        "server": {
            "cpu_percent": cpu_usage,
            "ram_percent": ram_usage,
            "ram_total_gb": round(ram.total / (1024**3), 2),
            "ram_note": ram_note,
            "disk_percent": disk_usage,
            "disk_total_gb": disk_total_gb,
            "disk_free_gb": disk_free_gb
        },
        "database": {
            "status": db_status,
            "latency_ms": db_latency,
            "total_users": total_users,
            "total_threads": total_threads
        },
        "qdrant": {
            "status": qdrant_status,
            "latency_ms": qdrant_latency,
            "vector_count": vector_count
        },
        "services": {
            "inngest_url": inngest_dashboard_url,
            "langsmith_url": "https://smith.langchain.com/",
            "render_url": "https://dashboard.render.com/"
        }
    }

@app.post("/api/login")
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(
            status_code=401, 
            detail="Error on Username or Password!"
        )

    access_token = create_access_token(
        data={
            "sub": db_user.username, 
            "role": db_user.role
        }
    )
        
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "username": db_user.username,
        "role": db_user.role 
    }

@app.get("/api/threads/{username}", dependencies=[Depends(get_current_user)])
def get_threads(username: str, db: Session = Depends(get_db)):
    threads = db.query(ChatThread).filter(ChatThread.username == username).order_by(ChatThread.updated_at.asc()).all()
    return {t.thread_id: t.title for t in threads}

@app.post("/api/threads")
def save_thread(thread: ThreadCreate, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    if thread.username != current_user.get("sub"): 
        raise HTTPException(status_code=403, detail="Forbidden")
    
    db_thread = db.query(ChatThread).filter(ChatThread.thread_id == thread.thread_id).first()
    if db_thread:
        db_thread.title = thread.title 
    else:
        new_thread = ChatThread(username=thread.username, thread_id=thread.thread_id, title=thread.title)
        db.add(new_thread) 
    db.commit()
    return {"message": "Thread saved successfully"}

@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    if thread.username != current_user.get("sub"): 
        raise HTTPException(status_code=403, detail="Forbidden")
    
    db_thread = db.query(ChatThread).filter(ChatThread.thread_id == thread_id).first()
    if db_thread:
        db.delete(db_thread)
        db.commit()
    return {"message": "Thread deleted successfully"}

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)