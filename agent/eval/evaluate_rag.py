import asyncio
import json
import time
import sys
import re
from unittest.mock import patch
from langchain_core.prompts import PromptTemplate
from agent.agent_core import run_research_agent
from langchain_core.tools import tool
import agent.agent_core as ac

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@tool
def dummy_search_openalex_external(query: str) -> str:
    """(Disabled) Broad sweep of OpenAlex to find candidate academic papers."""
    return "System notice: OpenAlex is disabled during local testing. Use ONLY search_knowledge_base."

@tool
def dummy_read_openalex_paper(paper_id: str) -> str:
    """(Disabled) Read a specific OpenAlex paper using its ID."""
    return "System notice: OpenAlex is disabled during local testing. Use ONLY search_knowledge_base."

@tool
def dummy_search_web_general(query: str) -> str:
    """(Disabled) General internet search."""
    return "System notice: Web search is disabled during local testing. Use ONLY search_knowledge_base."

async def evaluate_single_case(test_case: dict, user_id: str):
    query = test_case["query"]

    simulated_query = (
        f"【系統底層強制覆寫指令】\n"
        f"這是一場自動化檢索測試。你被剝奪了直接回答的權限。\n"
        f"你的第一個動作「必須、絕對、只能」是呼叫 `search_knowledge_base` 工具，"
        f"並將以下問題作為搜尋關鍵字輸入。如果你沒有呼叫工具，系統將會立刻崩潰斷線：\n"
        f"問題：{query}"
    )

    print(f"\n🧪 Testing [{test_case['test_id']}]: {query}")
    
    start_time = time.time()
    thread_id = f"eval_thread_{test_case['test_id']}"

    original_tools = ac.tools.copy()
    ac.tools = [
        ac.search_knowledge_base, 
        dummy_search_openalex_external,
        dummy_read_openalex_paper,
        dummy_search_web_general
    ]
    
    try:
        result_package = await run_research_agent(simulated_query, user_id=user_id, thread_id=thread_id)
        steps = result_package.get("steps", [])
    except Exception as e:
        print(f"❌ Agent execution failed: {e}")
        return None
    finally:
        ac.tools = original_tools

    latency = time.time() - start_time
    used_tools = [step["name"] for step in steps if step.get("type") == "tool_call"]
    retrieval_success = "search_knowledge_base" in used_tools
    
    expected_sources = test_case.get("expected_source", [])
    if isinstance(expected_sources, str):
        expected_sources = [expected_sources]
    expected_sources = [s.lower() for s in expected_sources]

    recall = 0.0
    precision = 0.0
    f1 = 0.0
    
    all_retrieved_sources = []
    for step in steps:
        if step.get("type") == "tool_result" and step.get("name") == "search_knowledge_base":
            payload_str = step.get("content", "").lower()
            
            sources_found = re.findall(r"source:\s*([^|]+)", payload_str)
            extracted = [s.strip() for s in sources_found]
            
            if extracted:
                all_retrieved_sources.extend([s.replace("temp_uploads/William", "").lower() for s in extracted])
            else:
                all_retrieved_sources.append(payload_str)
    
    unique_sources = set(all_retrieved_sources)
    print(f"Target: {expected_sources}")
    print(f"RAG: {unique_sources}")

    recall = 0.0
    precision = 0.0
    f1 = 0.0
    
    if expected_sources:
        if unique_sources:
            hits = sum(1 for exp_s in expected_sources if any(exp_s in act_s for act_s in unique_sources))
            
            recall = hits / len(expected_sources)     
            precision = hits / len(unique_sources) 
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
        else:
            print(f"⚠️ [檢索失敗] 預期應命中 {expected_sources}，但未撈出任何結果！")
            
    else:
        if unique_sources:
            print(f"⚠️ [過度檢索] 問題不在資料庫中，但系統卻強制撈出了雜訊")
            recall = 0.0
            precision = 0.0
            f1 = 0.0
        else:
            recall = 1.0
            precision = 1.0
            f1 = 1.0

    print(f"⏱️ Latency: {latency:.2f}s | 🛠️ Tools: {used_tools}")
    print(f"🎯 Recall: {recall:.2f} | Precision: {precision:.2f} | F1 Score: {f1:.2f}")
    
    return {
        "test_id": test_case["test_id"],
        "latency_sec": latency,
        "retrieval_success": retrieval_success,
        "recall": recall,
        "precision": precision,
        "f1": f1
    }

async def run_evaluation_suite(json_path: str, user_id: str):
    print(f"🚀 Starting Retrieval Evaluation Suite from {json_path}...")
    print(f"🛡️ OpenAlex & Web Tools disabled. LLM-Judge disabled.\n")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    results = []
    skipped_count = 0  # 🌟 新增：記錄被略過的題目數量
    
    for test_case in dataset:
        res = await evaluate_single_case(test_case, user_id)
        if res:
            # 🌟 新增：只將有呼叫工具的結果加入最終統計
            if res["retrieval_success"]:
                results.append(res)
            else:
                skipped_count += 1
                print(f"⏭️ [排除統計] 測試 [{res['test_id']}] Agent 未使用工具，不列入平均計算。\n")
            
    if results:
        avg_recall = sum(r["recall"] for r in results) / len(results)
        avg_precision = sum(r["precision"] for r in results) / len(results)
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        avg_latency = sum(r["latency_sec"] for r in results) / len(results)
        
        print("\n" + "="*45)
        print("🏆 RETRIEVAL PERFORMANCE REPORT 🏆")
        print("="*45)
        print(f"Total Cases in Dataset: {len(dataset)}")
        print(f"Valid Tests (Used Tool):{len(results)}")
        print(f"Skipped Tests:          {skipped_count}")
        print("-" * 45)
        print(f"Average Recall:     {avg_recall:.2f}")
        print(f"Average Precision:  {avg_precision:.2f}")
        print(f"Average F1 Score:   {avg_f1:.2f}")
        print(f"Average Latency:    {avg_latency:.2f} sec")
        print("="*45)
    else:
        print("\n❌ 所有測試皆未使用工具，無法計算平均成績。")

if __name__ == "__main__":
    asyncio.run(run_evaluation_suite("agent/eval/qa_dataset.json", user_id="William"))