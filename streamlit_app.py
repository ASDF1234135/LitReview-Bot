import streamlit as st
import requests
import uuid
import os
import json

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
SESSIONS_FILE = "user_sessions.json"

st.set_page_config(page_title="AI Research Agent", page_icon="🤖", layout="wide")


if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

if not st.session_state["authentication_status"]:
    st.title("🤖 AI Research Agent")
    st.write("Please login for the service")
    
    # 使用 Tabs 切換登入與註冊
    tab_login, tab_register = st.tabs(["Login", "Register"])
    
    with tab_login:
        with st.form("login_form"):
            login_user = st.text_input("Username")
            login_pass = st.text_input("Password", type="password")
            submitted_login = st.form_submit_button("Login", use_container_width=True)
            
            if submitted_login:
                if not login_user or not login_pass:
                    st.warning("Please fill the username and password")
                else:
                    res = requests.post(f"{API_BASE_URL}/api/login", json={
                        "username": login_user,
                        "password": login_pass
                    })
                    if res.status_code == 200:
                        st.session_state["authentication_status"] = True
                        st.session_state["username"] = res.json()["username"]
                        st.success("Login successful, loading...")
                        st.rerun()
                    else:
                        st.error(f"login failed: {res.json().get('detail', 'unknown error')}")
                        
    with tab_register:
        with st.form("register_form"):
            reg_user = st.text_input("Username")
            reg_email = st.text_input("Email")
            reg_pass = st.text_input("Password", type="password")
            submitted_reg = st.form_submit_button("Register", use_container_width=True)
            
            if submitted_reg:
                if not reg_user or not reg_email or not reg_pass:
                    st.warning("Please fill all the columns")
                else:
                    res = requests.post(f"{API_BASE_URL}/api/register", json={
                        "username": reg_user,
                        "email": reg_email,
                        "password": reg_pass
                    })
                    if res.status_code == 200:
                        st.success("Register successful! Please login")
                    else:
                        st.error(f"Register failed: {res.json().get('detail', 'unknown error')}")
elif st.session_state["authentication_status"]:
    USER_ID = st.session_state["username"]
    
    with st.sidebar:
        st.write(f'Welcome, **{USER_ID}**')
        if st.button("Logout", use_container_width=True):
            st.session_state["authentication_status"] = False
            st.session_state["username"] = None
            st.rerun()
        st.divider()


    def extract_text(content):
        if isinstance(content, str): return content
        if isinstance(content, list):
            return "".join([b.get("text", "") for b in content if isinstance(b, dict)])
        if isinstance(content, dict): return content.get("answer", str(content))
        return str(content)

    def load_threads(user_id):
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, "r") as f:
                data = json.load(f)
                user_data = data.get(user_id, {})
                if isinstance(user_data, list):
                    return {tid: "New Chat" for tid in user_data}
                return user_data
        return {}

    def save_threads(user_id, threads_dict):
        data = {}
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, "r") as f:
                data = json.load(f)
        data[user_id] = threads_dict
        with open(SESSIONS_FILE, "w") as f:
            json.dump(data, f)

    # ==========================================
    # 2. 狀態初始化 (修復缺失的 messages 陣列)
    # ==========================================
    if "threads" not in st.session_state:
        user_threads = load_threads(USER_ID)
        if not user_threads:
            new_tid = f"Session_{uuid.uuid4().hex[:5]}"
            user_threads = {new_tid: "New Chat"}
            save_threads(USER_ID, user_threads)
        st.session_state.threads = user_threads

    if "current_thread_id" not in st.session_state:
        st.session_state.current_thread_id = list(st.session_state.threads.keys())[-1]

    # 👉 [補回] 初始化對話紀錄與最後載入的 Thread
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_loaded_thread" not in st.session_state:
        st.session_state.last_loaded_thread = None

    def fetch_history(thread_id):
        try:
            res = requests.get(f"{API_BASE_URL}/api/history/{thread_id}")
            if res.status_code == 200:
                return res.json().get("history", [])
        except:
            pass
        return []

    if st.session_state.last_loaded_thread != st.session_state.current_thread_id:
        with st.spinner("Loading history from database..."):
            st.session_state.messages = fetch_history(st.session_state.current_thread_id)
            st.session_state.last_loaded_thread = st.session_state.current_thread_id

    @st.dialog("📂 Knowledge Base Management", width="large")
    def file_management_dialog():
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
                            st.success("✅ Ingestion started! Click Refresh in a few seconds.")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Failed to connect to backend: {e}")

        st.divider()
        
        col1, col2 = st.columns([4, 1])
        col1.subheader("📄 My Uploaded Files")
        if col2.button("🔄 Refresh", use_container_width=True):
            st.rerun() 
            
        try:
            files_res = requests.get(f"{API_BASE_URL}/api/files/{USER_ID}")
            if files_res.status_code == 200:
                files = files_res.json().get("files", [])
                if not files:
                    st.info("No files uploaded yet.")
                else:
                    for f in files:
                        fc1, fc2 = st.columns([5, 1])
                        fc1.write(f"📄 {f}")
                        if fc2.button("🗑️", key=f"del_{f}", help="Delete this file"):
                            requests.delete(f"{API_BASE_URL}/api/files/{USER_ID}/{f}")
                            st.rerun()
            else:
                st.error("Could not fetch files.")
        except:
            st.error("Backend offline.")

    # ==========================================
    # 4. 全新側邊欄 (Sidebar)
    # ==========================================
    with st.sidebar:
        st.header("💬 Chat Sessions")
        
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            new_tid = f"Session_{uuid.uuid4().hex[:5]}"
            st.session_state.threads[new_tid] = "New Chat"
            save_threads(USER_ID, st.session_state.threads)
            st.session_state.current_thread_id = new_tid
            st.rerun()

        st.divider()

        # 👉 垂直對話列表 + "..." 選單
        for tid, title in reversed(st.session_state.threads.items()):
            col1, col2 = st.columns([8, 2]) 
            
            is_current = (tid == st.session_state.current_thread_id)
            btn_type = "primary" if is_current else "secondary"

            with col1:
                if st.button(title, key=f"sel_{tid}", use_container_width=True, type=btn_type):
                    st.session_state.current_thread_id = tid
                    st.rerun()
                    
            with col2:
                with st.popover("⋮", use_container_width=True):
                    new_title = st.text_input("Rename", value=title, key=f"ren_{tid}", label_visibility="collapsed")
                    
                    if st.button("💾 Save", key=f"save_{tid}", use_container_width=True):
                        st.session_state.threads[tid] = new_title
                        save_threads(USER_ID, st.session_state.threads)
                        st.rerun()
                        
                    if st.button("🗑️ Delete", key=f"del_chat_{tid}", type="primary", use_container_width=True):
                        if len(st.session_state.threads) > 1:
                            del st.session_state.threads[tid]
                            save_threads(USER_ID, st.session_state.threads)
                            if st.session_state.current_thread_id == tid:
                                st.session_state.current_thread_id = list(st.session_state.threads.keys())[-1]
                            st.rerun()
                        else:
                            st.error("Cannot delete the last chat!")

        st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True) 
        if st.sidebar.button("📂 Manage Knowledge Base", use_container_width=True):
            file_management_dialog() 

    # ==========================================
    # 5. 主畫面 Chat Interface (結合 Streaming)
    # ==========================================
    st.title("🤖 Autonomous Research Agent")
    st.caption(f"Current Thread: `{st.session_state.current_thread_id}`")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ex: What is Active Learning?"):
        
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            status_container = st.status("🧠 Agent is analysing...", expanded=True)
            answer_placeholder = st.empty() 
            
            try:
                payload = {
                    "message": user_query,
                    "user_id": USER_ID,
                    "thread_id": st.session_state.current_thread_id
                }
                
                response = requests.post(f"{API_BASE_URL}/api/chat", json=payload, stream=True)
                
                if response.status_code == 200:
                    final_answer = ""
                    
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            
                            if data["type"] == "status":
                                status_container.write(data["content"])
                                
                            elif data["type"] == "answer":
                                final_answer = extract_text(data["content"])
                                
                    status_container.update(label="✅ Research Completed！", state="complete", expanded=False)
                    answer_placeholder.markdown(final_answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer
                    })
                    
                else:
                    status_container.update(label="❌ Error!", state="error")
                    st.error(f"Backend Error: {response.text}")
                    
            except Exception as e:
                status_container.update(label="❌ Connection Failed", state="error")
                st.error(f"Failed to reach backend: {e}")