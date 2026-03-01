import uuid
from locust import FastHttpUser, task, between

class RAGAgentUser(FastHttpUser):
    wait_time = between(5, 15)

    def on_start(self):
        # 只保留 user_id，移除 on_start 裡的 thread_id
        self.user_id = "Locust_Tester"

    @task
    def chat_with_agent(self):
        # 確保每次發送請求，都開啟一個全新的 Thread
        current_thread_id = f"TestSession_{uuid.uuid4().hex[:8]}"
        
        payload = {
            "message": "What is Active Learning? Explain based on the provided local documents.",
            "user_id": self.user_id,
            "thread_id": current_thread_id
        }

        with self.client.post("/api/chat", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP Error {response.status_code}: {response.text}")