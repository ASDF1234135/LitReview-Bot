import uuid
from locust import FastHttpUser, task, between

class RAGAgentUser(FastHttpUser):
    wait_time = between(5, 15)

    def on_start(self):
        self.username = f"locust_{uuid.uuid4().hex[:6]}"
        self.password = "loadtest123"

        self.client.post("/api/register", json={
            "username": self.username,
            "email": f"{self.username}@test.com",
            "password": self.password
        })

        with self.client.post("/api/login", json={
            "username": self.username,
            "password": self.password
        }, catch_response=True) as response:
            if response.status_code == 200:
                self.token = response.json().get("access_token")
            else:
                response.failure(f"Login failed: {response.text}")
                self.token = None

    @task
    def chat_with_agent(self):
        if not self.token:
            return

        current_thread_id = f"TestSession_{uuid.uuid4().hex[:8]}"
        
        payload = {
            "message": "Give me the detail review of mirror therapy",
            "user_id": self.username, 
            "thread_id": current_thread_id
        }

        headers = {
            "Authorization": f"Bearer {self.token}"
        }

        with self.client.post("/api/chat", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP Error {response.status_code}: {response.text}")