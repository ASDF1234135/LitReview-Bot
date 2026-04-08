# agent_core.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
import os
import json
from DB.database import SessionLocal, User
from pydantic import BaseModel, Field

from agent.tools import search_knowledge_base, search_openalex_external, read_openalex_paper, search_web_general
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

tools = [search_knowledge_base, search_openalex_external, read_openalex_paper, search_web_general]

SYSTEM_PROMPT = """
You are an elite AI Research Director managing a professional literature review process.

**Hierarchy of Information Trust (Strictly Enforced):**
1. **PRIVATE KNOWLEDGE** (`search_knowledge_base`): Your absolute ground truth and highest priority.
2. **ARXIV PAPERS** (`search_openalex_external` & `read_openalex_paper`): Highly trusted academic extensions. Use this to fill gaps or find state-of-the-art updates.
3. **WEB SEARCH** (`search_web_general`): Lowest trust. Use ONLY for broad concepts, general tech news, or non-academic context.

**Your Tool Arsenal:**
1. `search_knowledge_base`: Search the user's PRIVATE uploaded documents.
2. `search_openalex_external`: Broad sweep of OpenAlex to find candidate academic papers (returns abstracts and OpenAlex IDs).
3. `read_openalex_paper`: Spawns a sub-agent to completely download and deeply read a specific OpenAlex paper using its ID.
4. `search_web_general`: General internet search.

**Tool Parameter Rules (CRITICAL):**
The user prompt may contain an "Original Query" and sometimes an "Optimized Search Terms" or "Hypothetical Context for Search". You must adapt your tool inputs accordingly:
- For `search_knowledge_base`: You may pass the full "Hypothetical Context" or long descriptive paragraphs. The vector database handles long text well.
- For `search_openalex_external`: NEVER pass long paragraphs or hypothetical context directly. You MUST extract ONLY 2 to 4 highly specific academic keywords (e.g., "Generative Active Learning LLM") to use as the search query.

**Strict Rules of Engagement:**
1. **The Research Pipeline:** ALWAYS follow this logical flow unless instructed otherwise:
   - **Step 1 (Grounding):** ALWAYS start by using `search_knowledge_base` to check what the user already knows, what their current projects are, or what private data they have on the topic.
   - **Step 2 (Extension):** If the private knowledge is insufficient, or the user explicitly asks for new external literature, use `search_openalex_external` to gather candidate paper IDs.
   - **Step 3 (Deep Dive):** Identify the most critical 1-3 papers from the OpenAlex sweep, and use `read_openalex_paper` to extract deep methodologies, hyperparameters, limitations, or data.
   - **Step 4 (Synthesis):** Cross-reference the external OpenAlex findings with the baseline established from the private knowledge base. Clearly state how the new literature relates to or fills the gaps in the user's private documents.
2. **Strict Domain Boundaries:** You are an Academic Research Tool, not a general-purpose chatbot. You are explicitly FORBIDDEN from engaging in casual conversation, creative writing, programming tutorials, or answering general knowledge questions.
Evaluate EVERY prompt before acting. If the prompt is not a direct inquiry about academic literature or scientific methodologies, you MUST abort all operations and output exactly: "Error: Query falls outside the permitted academic research domain."
3. **Do Not Hallucinate:** If a paper's details are not in the abstract, you MUST use `read_openalex_paper` to find out. Do not guess methodologies.
4. **Do Not Use Internal Knowledge:** Answer strictly and exclusively based on the evidence returned by your tools. Your pre-trained knowledge is strictly forbidden. If the search results lack the direct proof needed to answer the query, you must answer "I don't know" without attempting to guess or infer.
5. **Citation:** Always cite sources precisely. 
   - For private docs: "[<Author(s)>, <Year>. <Title> | Source: Local_Knowledge_base]"
   - For OpenAlex: "[<Author(s)>, <Year>. <Title> | Source: OpenAlex]"
   - For Web: "[Web: <URL>]"
   - You need to provide a completed reference list at the end of the response when the numbers of referece are over 5.
"""

DB_URI = os.getenv("POSTGRES_URI", "postgresql://admin:secretpassword@localhost:5432/agent_memory")


class RouteDecision(BaseModel):
    strategy: str = Field(description="The chosen strategy: 'direct', 'rewrite', or 'hyde'")
    reasoning: str = Field(description="Brief explanation of why this strategy was chosen")

router_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
router_chain = router_llm.with_structured_output(RouteDecision) 

async def analyze_query_intent(query: str) -> RouteDecision:
    prompt = f"""
    You are an expert intent classifier for an Academic AI Research Assistant.
    Analyze the user's query and route it to the MOST EFFICIENT retrieval strategy.

    1. "direct": Use when the query contains highly specific keywords, proper nouns, algorithm names (e.g., PEGASUS, ALScope), or asks for specific factual numbers.
    2. "rewrite": Use when the query is a comparison, contains multiple sub-questions, or is grammatically conversational and needs translation into formal search queries.
    3. "hyde": Use when the query is vague, describes a concept or symptom without knowing the exact terminology, or asks about "a paper that does X" without naming it.

    User Query: {query}
    """
    try:
        return await router_chain.ainvoke(prompt)
    except Exception as e:
        print(f"[Router Error] Defaulting to direct. Error: {e}")
        return RouteDecision(strategy="direct", reasoning="Fallback due to error")

async def rewrite_query(query: str) -> str:
    prompt = f"""
    Rewrite the following conversational or complex user query into a concise, 
    highly effective list of academic search keywords. Focus on core entities and methodologies.
    Query: {query}
    Output ONLY the keywords, separated by spaces.
    """
    res = await router_llm.ainvoke(prompt)
    return res.content.strip()

async def generate_hyde_document(query: str) -> str:
    prompt = f"""
    Write a short, hypothetical academic paragraph that answers the following query. 
    Use formal scientific terminology, likely methodologies, and academic phrasing. 
    Do not use introductory filler (e.g., "Here is a paragraph").
    Query: {query}
    """
    res = await router_llm.ainvoke(prompt)
    return res.content.strip()

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

    yield json.dumps({"type": "status", "content": "Analysing Question..."}) + "\n"
    
    decision = await analyze_query_intent(user_input)
    strategy = decision.strategy.lower()
    print(f"--- [Router] Strategy: {strategy} | Reason: {decision.reasoning} ---")
    
    final_input = user_input
    
    if strategy == "rewrite":
        yield json.dumps({"type": "status", "content": "Transform to Acadamical Question..."}) + "\n"
        optimized_query = await rewrite_query(user_input)
        final_input = f"Original Query: {user_input}\nOptimized Search Terms: {optimized_query}"
        
    elif strategy == "hyde":
        yield json.dumps({"type": "status", "content": "HyDE Activated..."}) + "\n"
        hyde_doc = await generate_hyde_document(user_input)
        final_input = f"Original Query: {user_input}\nHypothetical Context for Search: {hyde_doc}"

    inputs = {
        "messages": [("user", f"User ID: {user_id}\nRequest: {final_input}")]
    }

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        
        agent_executor = create_react_agent(
            llm, 
            tools, 
            checkpointer=checkpointer,
            prompt=SYSTEM_PROMPT
        )

        async for event in agent_executor.astream(inputs, config=config):
            if "agent" in event:
                message = event["agent"]["messages"][-1]
                if message.tool_calls:
                    for tc in message.tool_calls:
                        yield json.dumps({
                            "type": "status", 
                            "content": f"Agent is using tool: `{tc['name']}`..."
                        }) + "\n"
                else:
                    current_tokens = 0
                    
                    if hasattr(message, "usage_metadata") and message.usage_metadata:
                        current_tokens = message.usage_metadata.get("total_tokens", 0)
                    elif hasattr(message, "response_metadata") and "token_usage" in message.response_metadata:
                        current_tokens = message.response_metadata["token_usage"].get("total_tokens", 0)
                    
                    if current_tokens > 0:
                        try:
                            with SessionLocal() as db:
                                user_record = db.query(User).filter(User.username == user_id).first()
                                if user_record:
                                    user_record.total_tokens = (user_record.total_tokens or 0) + current_tokens
                                    db.commit()
                                    print(f"--- [Billing] User '{user_id}' consumed {current_tokens} tokens. ---")
                        except Exception as e:
                            print(f"--- [Billing Error] Failed to update token usage: {e} ---")


                    yield json.dumps({
                        "type": "answer", 
                        "content": message.content
                    }) + "\n"

            elif "tools" in event:
                for message in event["tools"]["messages"]:
                    yield json.dumps({
                        "type": "status", 
                        "content": f"Tool `{message.name}` completed, now reading..."
                    }) + "\n"

async def run_research_agent(user_input: str, user_id: str, thread_id: str = "1"):
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        }
    }
    decision = await analyze_query_intent(user_input)
    strategy = decision.strategy.lower()
    print(f"--- [Router] Strategy: {strategy} | Reason: {decision.reasoning} ---")
    
    final_input = user_input
    
    if strategy == "rewrite":
        optimized_query = await rewrite_query(user_input)
        final_input = f"Original Query: {user_input}\nOptimized Search Terms: {optimized_query}"
        
    elif strategy == "hyde":
        hyde_doc = await generate_hyde_document(user_input)
        final_input = f"Original Query: {user_input}\nHypothetical Context for Search: {hyde_doc}"

    inputs = {
        "messages": [("user", f"User ID: {user_id}\nRequest: {final_input}")]
    }

    result_package = {"answer": "", "steps": []}

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        
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
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()
            
            agent_executor = create_react_agent(
                llm, 
                tools, 
                checkpointer=checkpointer,
                prompt=SYSTEM_PROMPT
            )
            
            state = await agent_executor.aget_state(config)
            
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
                    
                    if isinstance(content, list):
                        content = "".join([b.get("text", "") for b in content if isinstance(b, dict)])
                    
                    if isinstance(content, str) and content.strip():
                        formatted_history.append({"role": "assistant", "content": content})
                        
            return formatted_history
            
    except Exception as e:
        print(f"--- [Error] Failed to fetch history for {thread_id}: {e} ---")
        return []