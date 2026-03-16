# agent_core.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
import os
import json

from tools import search_knowledge_base, search_arxiv_external, read_arxiv_paper, search_web_general

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

tools = [search_knowledge_base, search_arxiv_external, read_arxiv_paper, search_web_general]

SYSTEM_PROMPT = """
You are an elite AI Research Director managing a literature review process.

**Your Tool Arsenal:**
1. `search_knowledge_base`: Search the user's PRIVATE uploaded documents. (Highly trusted).
2. `search_arxiv_external`: Broad sweep of ArXiv to find candidate papers (returns abstracts and IDs).
3. `read_arxiv_paper`: Spawns a sub-agent to deeply read a specific ArXiv paper using its ID.
4. `search_web_general`: General internet search for history, tech news, or broad concepts.

**Strict Rules of Engagement:**
1. **The Research Pipeline:** If asked to review external literature, ALWAYS follow this pipeline:
   - Step 1: Use `search_arxiv_external` to gather a list of candidate paper IDs.
   - Step 2: Identify the most critical 1 to 3 papers from the sweep.
   - Step 3: Use `read_arxiv_paper` on those specific IDs to extract deep methodologies, limitations, or data.
   - Step 4: If the user mentions their own project/documents, use `search_knowledge_base` to cross-reference the external findings with their private data.
2. **Do Not Hallucinate:** If a paper's details are not in the abstract, you MUST use `read_arxiv_paper` to find out. Do not guess methodologies.
3. **Citation:** Always cite sources precisely. 
   - For private docs: "[Source: <filename>, Title: <title>]"
   - For ArXiv: "[ArXiv: <ID>, Year: <year>]"
"""

DB_URI = os.getenv("POSTGRES_URI", "postgresql://admin:secretpassword@localhost:5432/agent_memory")

async def run_research_agent_stream(user_input: str, user_id: str, thread_id: str = "1"):
    """
    串流版本的 Agent 執行函數。
    每當 Agent 執行一個動作，就立刻 yield 出狀態字串。
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    }
    inputs = {
        "messages": [("user", f"User ID: {user_id}\nRequest: {user_input}")]
    }

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        
        agent_executor = create_react_agent(
            llm, 
            tools, 
            checkpointer=checkpointer,
            prompt=SYSTEM_PROMPT
        )

        # astream 會在每個 Node 執行完畢時觸發
        async for event in agent_executor.astream(inputs, config=config):
            if "agent" in event:
                message = event["agent"]["messages"][-1]
                if message.tool_calls:
                    # Agent 決定呼叫工具
                    for tc in message.tool_calls:
                        yield json.dumps({
                            "type": "status", 
                            "content": f"🛠️ Agent is using tool: `{tc['name']}`..."
                        }) + "\n"
                else:
                    # Agent 給出最終回答
                    yield json.dumps({
                        "type": "answer", 
                        "content": message.content
                    }) + "\n"

            elif "tools" in event:
                # 工具執行完畢
                for message in event["tools"]["messages"]:
                    yield json.dumps({
                        "type": "status", 
                        "content": f"✅ Tool `{message.name}` completed, now reading..."
                    }) + "\n"

async def run_research_agent(user_input: str, user_id: str, thread_id: str = "1"):
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    }
    inputs = {
        "messages": [("user", f"User ID: {user_id}\nRequest: {user_input}")]
    }
    result_package = {"answer": "", "steps": []}

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup() # 確保表格存在
        
        agent_executor = create_react_agent(
            llm, 
            tools, 
            checkpointer=checkpointer,
            prompt=SYSTEM_PROMPT
        )

        async for event in agent_executor.astream(inputs, config=config):
            for key, value in event.items():
                if key == "agent":
                    message = value["messages"][-1]
                    if message.tool_calls:
                        for tc in message.tool_calls:
                            result_package["steps"].append({
                                "type": "tool_call",
                                "name": tc["name"],
                                "args": tc["args"],
                                "id": tc["id"]
                            })
                    else:
                        result_package["answer"] = message.content

                elif key == "tools":
                    for message in value["messages"]:
                        result_package["steps"].append({
                            "type": "tool_result",
                            "name": message.name,
                            "content": message.content,
                            "tool_call_id": message.tool_call_id
                        })

    return result_package

async def get_thread_history(thread_id: str):
    """從 Postgres 讀取指定 Thread 的歷史對話紀錄"""
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # [關鍵修正] 在這裡建立資料庫連線與 Agent 實例
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            # 建立一個臨時的 Agent Executor 來幫我們撈取狀態
            agent_executor = create_react_agent(
                llm, 
                tools, 
                checkpointer=checkpointer,
                prompt=SYSTEM_PROMPT
            )
            
            state = await agent_executor.aget_state(config)
            
            # 如果找不到這個 thread_id 的狀態，代表是新對話
            if not state or not hasattr(state, "values"):
                return []
                
            messages = state.values.get("messages", [])
            formatted_history = []
            
            for msg in messages:
                if msg.type == "human":
                    content = msg.content
                    # 過濾掉我們在後端偷偷塞入的 "User ID: xxx\nRequest: " 字串
                    if isinstance(content, str) and "Request: " in content:
                        content = content.split("Request: ", 1)[-1]
                    formatted_history.append({"role": "user", "content": content})
                    
                elif msg.type == "ai" and msg.content:
                    content = msg.content
                    
                    # 處理 Gemini 可能回傳 List Block 的情況
                    if isinstance(content, list):
                        content = "".join([b.get("text", "") for b in content if isinstance(b, dict)])
                    
                    # 確保真的有文字才加入
                    if isinstance(content, str) and content.strip():
                        formatted_history.append({"role": "assistant", "content": content})
                        
            return formatted_history
            
    except Exception as e:
        print(f"--- [Error] Failed to fetch history for {thread_id}: {e} ---")
        return []