import streamlit as st
import requests
import uuid
import os
import json

# --- 設定 ---
API_BASE_URL = "http://localhost:8000"
USER_ID = "User_William" 
SESSIONS_FILE = "user_sessions.json" # 用來記憶 thread_ids 的本地小檔案

st.set_page_config(page_title="AI Research Agent", page_icon="🤖", layout="wide")

def extract_text(content):
    if isinstance(content, str): return content
    if isinstance(content, list):
        return "".join([b.get("text", "") for b in content if isinstance(b, dict)])
    if isinstance(content, dict): return content.get("answer", str(content))
    return str(content)

# --- 讀寫 Session 檔案的 Helper ---
def load_thread_ids(user_id):
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r") as f:
            data = json.load(f)
            # 只回傳屬於這個 user_id 的對話列表
            return data.get(user_id, [])
    return []

def save_thread_ids(user_id, thread_ids):
    data = {}
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r") as f:
            data = json.load(f)
            
    # 更新該 user_id 的對話列表
    data[user_id] = thread_ids
    
    with open(SESSIONS_FILE, "w") as f:
        json.dump(data, f)

# --- 狀態初始化 ---
if "thread_ids" not in st.session_state:
    # 載入時傳入目前的 USER_ID
    ids = load_thread_ids(USER_ID)
    if not ids:
        ids = [f"Session_{uuid.uuid4().hex[:5]}"]
        save_thread_ids(USER_ID, ids)
    st.session_state.thread_ids = ids

if "current_thread_id" not in st.session_state:
    # 預設載入最後一個對話
    st.session_state.current_thread_id = st.session_state.thread_ids[-1]

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 偵測切換對話：向後端拉取歷史紀錄 ---
def fetch_history(thread_id):
    try:
        res = requests.get(f"{API_BASE_URL}/api/history/{thread_id}")
        if res.status_code == 200:
            return res.json().get("history", [])
    except:
        pass
    return []

# 如果發現當前 thread 改變了，或者剛重啟，就去後端拉資料
if st.session_state.get("last_loaded_thread") != st.session_state.current_thread_id:
    with st.spinner("Loading history from database..."):
        st.session_state.messages = fetch_history(st.session_state.current_thread_id)
        st.session_state.last_loaded_thread = st.session_state.current_thread_id

# --- Sidebar ---
with st.sidebar:
    st.header("💬 Chat Sessions")
    if st.button("➕ New Chat", use_container_width=True):
        new_thread_id = f"Session_{uuid.uuid4().hex[:5]}"
        st.session_state.thread_ids.append(new_thread_id)
        save_thread_ids(USER_ID, st.session_state.thread_ids) # 存入本地檔案
        st.session_state.current_thread_id = new_thread_id
        st.rerun()

    selected_thread = st.radio(
        "History",
        options=reversed(st.session_state.thread_ids), # 反轉陣列，讓最新的在上面
        index=list(reversed(st.session_state.thread_ids)).index(st.session_state.current_thread_id),
        label_visibility="collapsed"
    )

    if selected_thread != st.session_state.current_thread_id:
        st.session_state.current_thread_id = selected_thread
        st.rerun()

    st.divider()
    st.header("📂 Knowledge Base")
    st.caption(f"Current User: {USER_ID}")
    
    uploaded_file = st.file_uploader("Upload PDF Paper", type=["pdf"])
    if uploaded_file:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if st.button("🚀 Process & Ingest", use_container_width=True):
            with st.spinner("Dispatching to background worker..."):
                try:
                    payload = {"file_path": file_path, "user_id": USER_ID}
                    res = requests.post(f"{API_BASE_URL}/api/trigger-ingest", json=payload)
                    if res.status_code == 200:
                        st.success("✅ Ingestion started!")
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

# --- Main Area ---
st.title("🤖 Autonomous Research Agent")
st.caption(f"Current Thread: `{st.session_state.current_thread_id}`")

# 1. 顯示歷史訊息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. 處理使用者輸入
if user_query := st.chat_input("Ex: What is Active Learning?"):
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent is researching..."):
            try:
                payload = {
                    "message": user_query,
                    "user_id": USER_ID,
                    "thread_id": st.session_state.current_thread_id
                }
                
                response = requests.post(f"{API_BASE_URL}/api/chat", json=payload)
                
                if response.status_code == 200:
                    api_data = response.json()
                    raw_answer = api_data.get("response", {}).get("answer", "No answer generated.")
                    clean_answer = extract_text(raw_answer)
                    
                    st.markdown(clean_answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": clean_answer
                    })
                else:
                    st.error(f"Backend Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to reach backend: {e}")