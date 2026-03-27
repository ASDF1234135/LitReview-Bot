import streamlit as st
import requests
import uuid
import os
import json

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
st.set_page_config(page_title="AI Research Agent", page_icon="🤖", layout="wide")

if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None

if not st.session_state["authentication_status"]:
    st.title("AI Research Agent")
    st.write("Please login for the service")
    
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
                        st.session_state["access_token"] = res.json()["access_token"]
                        st.session_state["role"] = res.json()["role"]
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

    def get_auth_headers():
        return {"Authorization": f"Bearer {st.session_state.get('access_token', '')}"}
    
    with st.sidebar:
        st.write(f'Welcome, **{USER_ID}**')
        if st.button("Logout", use_container_width=True):
            st.session_state.clear() 
            st.rerun()
        st.divider()


    def extract_text(content):
        if isinstance(content, str): return content
        if isinstance(content, list):
            return "".join([b.get("text", "") for b in content if isinstance(b, dict)])
        if isinstance(content, dict): return content.get("answer", str(content))
        return str(content)

    def load_threads_api(username):
        try:
            res = requests.get(f"{API_BASE_URL}/api/threads/{username}", headers=get_auth_headers())
            if res.status_code == 200:
                data = res.json()
                return data if data else {}
        except:
            pass
        return {}

    def save_thread_api(username, thread_id, title):
        requests.post(f"{API_BASE_URL}/api/threads", headers=get_auth_headers(), json={
            "username": username,
            "thread_id": thread_id,
            "title": title
        })
        
    def delete_thread_api(thread_id):
        requests.delete(f"{API_BASE_URL}/api/threads/{thread_id}", headers=get_auth_headers())

    # ==========================================
    # 2. 狀態初始化 (修復缺失的 messages 陣列)
    # ==========================================
    if "threads" not in st.session_state:
        user_threads = load_threads_api(USER_ID)
        if not user_threads:
            new_tid = f"Session_{uuid.uuid4().hex[:5]}"
            user_threads = {new_tid: "New Chat"}
            save_thread_api(USER_ID, new_tid, "New Chat")
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
            res = requests.get(f"{API_BASE_URL}/api/history/{thread_id}", headers=get_auth_headers())
            if res.status_code == 200:
                return res.json().get("history", [])
        except:
            pass
        return []

    if st.session_state.last_loaded_thread != st.session_state.current_thread_id:
        with st.spinner("Loading history from database..."):
            st.session_state.messages = fetch_history(st.session_state.current_thread_id)
            st.session_state.last_loaded_thread = st.session_state.current_thread_id

    @st.dialog("Knowledge Base Management", width="large")
    def file_management_dialog():
        st.caption(f"Current User: {USER_ID}")
        
        uploaded_files = st.file_uploader("Upload PDF Papers", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            if st.button(f"Process {len(uploaded_files)} Files", use_container_width=True):
                with st.spinner(f"Dispatching {len(uploaded_files)} files to background workers..."):
                    try:
                        files_payload = [
                            ("files", (file.name, file.getvalue(), "application/pdf")) 
                            for file in uploaded_files
                        ]
                        data = {"user_id": USER_ID}
                        
                        res = requests.post(f"{API_BASE_URL}/api/trigger-ingest", files=files_payload, data=data, headers=get_auth_headers())
                        
                        if res.status_code == 200:
                            st.success(f"Successfully queued {len(uploaded_files)} files! Click Refresh in a few seconds.")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Failed to connect to backend: {e}")

        st.divider()
        
        col1, col2 = st.columns([4, 1])
        col1.subheader("My Uploaded Files")
        
        col2.button("🔄 Refresh", use_container_width=True)
        def delete_file_callback(filename):
            requests.delete(f"{API_BASE_URL}/api/files/{USER_ID}/{filename}", headers=get_auth_headers())

        try:
            files_res = requests.get(f"{API_BASE_URL}/api/files/{USER_ID}", headers=get_auth_headers())
            if files_res.status_code == 200:
                files = files_res.json().get("files", [])
                if not files:
                    st.info("No files uploaded yet.")
                else:
                    for f in files:
                        fc1, fc2 = st.columns([5, 1])
                        fc1.write(f"📄 {f}")
                        fc2.button("🗑️", key=f"del_{f}", help="Delete this file", on_click=delete_file_callback, args=(f,))
            else:
                st.error("Could not fetch files.")
                
        except Exception as e:
            st.error("Backend offline.")

    # ==========================================
    # 4. 全新側邊欄 (Sidebar)
    # ==========================================
    with st.sidebar:

        if st.session_state.get("role") == "admin":
            st.caption("Admin Privileges")
            if st.button("Enter Admin Dashboard", type="primary"):
                st.session_state["view_mode"] = "admin"
                st.rerun()
            st.divider()
        
        st.header("💬 Chat Sessions")
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            new_tid = f"Session_{uuid.uuid4().hex[:5]}"
            st.session_state.threads[new_tid] = "New Chat"
            save_thread_api(USER_ID, new_tid, "New Chat")
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
                        save_thread_api(USER_ID, tid, new_title)
                        st.rerun()
                        
                    if st.button("🗑️ Delete", key=f"del_chat_{tid}", type="primary", use_container_width=True):
                        if len(st.session_state.threads) > 1:
                            del st.session_state.threads[tid]
                            delete_thread_api(tid)
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
    current_view = st.session_state.get("view_mode", "chat")

    if current_view == "chat":
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
                    
                    response = requests.post(f"{API_BASE_URL}/api/chat", json=payload, stream=True, headers=get_auth_headers())
                    
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

    elif current_view == "admin" and st.session_state.get("role") == "admin":
        st.title("🎛️ System Triage Control Center")
        if st.button("🔙 Back to Chat"):
            st.session_state["view_mode"] = "chat"
            st.rerun()
            
        st.divider()
        
        # 準備出示 JWT 通行證去敲後端的門
        headers = {
            "Authorization": f"Bearer {st.session_state.get('access_token')}"
        }
        
        with st.spinner("Pinging microservices..."):
            try:
                # 呼叫受 JWT 保護的 API
                res = requests.get(f"{API_BASE_URL}/api/admin/health", headers=headers)
                
                if res.status_code == 200:
                    data = res.json()
                    st.success("Authentication Passed!")
                    # 暫時先用 JSON 印出來，確認有接通
                    st.json(data)
                elif res.status_code == 401 or res.status_code == 403:
                    st.error("🚨 Unauthorized Access! Invalid token or insufficient permissions.")
                else:
                    st.error(f"Error: {res.status_code}")
                    
            except Exception as e:
                st.error("Backend offline or unreachable.")