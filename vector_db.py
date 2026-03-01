from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, SparseVectorParams, Distance, PointStruct
from flashrank import Ranker, RerankRequest
# [NEW] 引入 FastEmbed 用來算 BM25
from fastembed import SparseTextEmbedding
import uuid
import os

class QdrantStorage:
    def __init__(self, collection='research_papers', dim=3072):
        db_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection = collection
        
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        # 2. 加入重試機制 (Retry Mechanism)，容忍 Qdrant 開機延遲
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"--- [DB] Connecting to Qdrant at {db_url} (Attempt {attempt+1}/{max_retries}) ---")
                self.client = QdrantClient(url=db_url, timeout=30)
                
                # 測試連線並檢查 Collection
                if not self.client.collection_exists(self.collection):
                    self.client.create_collection(
                        collection_name=self.collection,
                        vectors_config={
                            "dense": VectorParams(size=dim, distance=Distance.COSINE)
                        },
                        sparse_vectors_config={
                            "sparse": SparseVectorParams(index=models.SparseIndexParams(
                                on_disk=False,
                            ))
                        }
                    )
                print("--- [DB] Qdrant connection successful! ---")
                break # 成功就跳出迴圈
                
            except Exception as e:
                print(f"--- [DB] Failed to connect to Qdrant: {e} ---")
                if attempt < max_retries - 1:
                    print("--- [DB] Waiting 3 seconds before retrying... ---")
                    time.sleep(3)
                else:
                    raise Exception("Critical: Could not connect to Qdrant after multiple attempts.")

    def upsert(self, texts: list, metadatas: list, vectors: list, user_id: str, access: str):
        # [NEW] 計算 Sparse Vectors
        print("--- [DB] Computing Sparse Vectors (BM25) ---")
        sparse_vectors = list(self.sparse_model.embed(texts))
        
        points = []
        for i, text in enumerate(texts):
            payload = metadatas[i].copy()
            payload.update({
                "text": text,
                "user_id": user_id,
                "access": access
            })
            
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                # [NEW] 同時存入兩種向量
                vector={
                    "dense": vectors[i],
                    "sparse": sparse_vectors[i].as_object() # 轉為 Qdrant 格式
                },
                payload=payload
            ))
        
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector, query_text, top_k=5, user_id=None, strategy="hybrid"):
        """
        真正的混合搜尋策略：
        1. Dense Search (語意)
        2. Sparse Search (BM25 關鍵字)
        3. RRF Fusion (合併結果)
        4. Rerank (最終精排)
        """
        
        query_filter = models.Filter(
            should=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="access", match=models.MatchValue(value="public"))
            ]
        )

        limit = top_k * 3 # 取多一點來做 Fusion

        # [NEW] 混合搜尋邏輯
        if strategy == "hybrid":
            # 1. 計算 Query 的 Sparse Vector
            query_sparse = list(self.sparse_model.embed([query_text]))[0]

            # 2. 使用 Prefetch 執行並行搜尋
            response = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    # A. 語意搜尋
                    models.Prefetch(
                        query=query_vector,
                        using="dense",
                        filter=query_filter,
                        limit=limit
                    ),
                    # B. 關鍵字搜尋 (BM25)
                    models.Prefetch(
                        query=query_sparse.as_object(),
                        using="sparse",
                        filter=query_filter,
                        limit=limit
                    ),
                ],
                # C. Fusion: 使用 RRF 合併排名
                query=models.FusionQuery(
                    method=models.Fusion.RRF
                ),
                limit=limit,
                with_payload=True
            )
        else:
            # 傳統 Dense Only
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                using="dense",
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )

        results = response.points
        
        # 格式化
        formatted_results = []
        for r in results:
            formatted_results.append({
                "id": r.id,
                "text": r.payload.get("text"),
                "metadata": {k:v for k,v in r.payload.items() if k not in ['text']}
            })

        # 最後還是要做一次 Rerank，因為 Fusion 只是粗排
        if formatted_results:
            print(f"--- [DB] Reranking {len(formatted_results)} candidates ---")
            passages = [{"id": r["id"], "text": r["text"]} for r in formatted_results]
            rerank_req = RerankRequest(query=query_text, passages=passages)
            ranked = self.ranker.rerank(rerank_req)
            
            ranked_ids = [r["id"] for r in ranked[:top_k]]
            final_results = [r for r in formatted_results if r["id"] in ranked_ids]
            final_results.sort(key=lambda x: ranked_ids.index(x["id"]))
            return final_results
            
        return formatted_results
    
    def get_user_files(self, user_id: str) -> list:
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
                ),
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            files = set()
            for r in records:
                # 抓取 metadata 中的 source (通常是檔案路徑)
                source = r.payload.get("source")
                if source:
                    files.add(os.path.basename(source)) # 只取檔名
            return list(files)
        except Exception as e:
            print(f"--- [DB Error] Failed to get files: {e} ---")
            return []

    # [NEW] 刪除使用者的特定檔案
    def delete_user_file(self, user_id: str, filename: str) -> bool:
        try:
            # 1. 先找出該使用者所有的 Chunk
            records, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
                ),
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            # 2. 篩選出檔名相符的 Point IDs
            ids_to_delete = []
            for r in records:
                source = r.payload.get("source", "")
                if os.path.basename(source) == filename:
                    ids_to_delete.append(r.id)
            
            # 3. 執行批次刪除
            if ids_to_delete:
                print(f"--- [DB] Deleting {len(ids_to_delete)} chunks for file: {filename} ---")
                self.client.delete(
                    collection_name=self.collection,
                    points_selector=models.PointIdsList(points=ids_to_delete)
                )
            return True
        except Exception as e:
            print(f"--- [DB Error] Failed to delete file: {e} ---")
            return False