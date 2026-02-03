import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from inngest.experimental.ai import gemini
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_chunk_pdf, embed_texts, chunk_text
from vector_db import QdrantStorage
from custom_type import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult
from agent_core import research_agent

load_dotenv()

inngest_client = inngest.Inngest(
    app_id='rag_app',
    logger=logging.getLogger('uvicorn'),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event='rag/ingest_pdf'),
    concurrency=[inngest.Concurrency(limit=5)]
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data['pdf_path']
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunk_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunk_and_src.chunks
        source_id = chunk_and_src.source_id
        vecs = embed_texts(chunks)
        user_id = ctx.event.data.get("user_id", "default_user")

        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{
                "source": source_id,
                "text": chunks[i],
                "user_id": user_id,
                "access": "private"
            } for i in range(len(chunks))]

        QdrantStorage().upsert(ids, vecs, payloads)

        return RAGUpsertResult(ingested=len(chunks))
        

    chunk_and_src = await ctx.step.run("load-and-chunk", lambda:_load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda:_upsert(chunk_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Research Agent",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data['question']
    
    async def run_agent_workflow():
        # initialize
        initial_state = {
            "question": question,
            "user_id": ctx.event.data.get("user_id", "default_user"), 
            "router_decision": "",
            "local_contexts": [],
            "external_contexts": [],
            "external_docs": [],
            "is_sufficient": False,
            "sources": [],
            "final_answer": "",
            "iteration_count": 0
        }
        
        # Start LangGraph
        final_state = await research_agent.ainvoke(initial_state)
        return final_state

    # Run Agent
    result_state = await ctx.step.run("agent-reasoning-loop", run_agent_workflow)

    # Process Rejection
    if result_state.get("router_decision") == "reject":
        return {
            "answer": "We're sorry, your request failed the security review or is not an academically related issue, "
                        "so the system refused to process it.",
            "sources": [],
            "num_contexts": 0
        }
    
    # external docs
    external_docs = result_state.get("external_docs", [])
    if external_docs:
        events = []
        current_user_id = result_state.get("user_id", "default_user")

        for doc in external_docs:
            events.append(inngest.Event(
                name="rag/ingest_external_doc",
                data={"document": doc, "user_id": current_user_id}
            ))
        
        if events:
            await ctx.step.send_event("trigger-auto-ingest", events=events)

    return {
        "answer": result_state.get("final_answer", "No answer generated."),
        "sources": result_state.get("sources", []),
        "num_contexts": len(result_state.get("local_contexts", []))
    }

@inngest_client.create_function(
    fn_id="RAG: Ingest External Doc",
    trigger=inngest.TriggerEvent(event="rag/ingest_external_doc"),
    concurrency=[inngest.Concurrency(limit=10)]
)
async def rag_ingest_external_doc(ctx: inngest.Context):
    doc = ctx.event.data['document']
    source_id = doc.get('url')
    text_content = doc.get('full_content', '')
    user_id = ctx.event.data.get("user_id", "system")
    
    if not text_content:
        return {"status": "skipped", "reason": "empty content"}

    chunks = chunk_text(text_content)
    
    vecs = embed_texts(chunks)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
    
    payloads = [{
        "source": source_id, 
        "text": chunks[i], 
        "title": doc.get('title'),
        "type": "external_arxiv",
        "ingest_type": doc.get('ingest_type'),
        "user_id": user_id,
        "access": "public"
    } for i in range(len(chunks))]

    QdrantStorage().upsert(ids, vecs, payloads)
    
    return {
        "status": "success", 
        "source": source_id, 
        "chunks": len(chunks),
        "ingest_type": doc.get('ingest_type')
    }



app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai, rag_ingest_external_doc])