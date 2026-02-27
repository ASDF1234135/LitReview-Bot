# locustfile.py
from locust import HttpUser, task, between
import random

class AgentUser(HttpUser):
    # 模擬每個用戶發出請求後的思考等待時間 (1 到 5 秒)
    wait_time = between(1, 5)

    @task
    def chat_with_agent(self):
        # 準備測試用的 Payload
        payload = {
            "message": "What is the definition of Active Learning?",
            "user_id": f"test_user_{random.randint(1, 100)}",
            "thread_id": f"thread_{random.randint(1, 1000)}"
        }
        
        # 打您的 FastAPI 端點
        with self.client.post("/api/chat", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status: {response.status_code}")