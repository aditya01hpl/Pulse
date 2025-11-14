from __future__ import annotations

import os
import random
from locust import HttpUser, task, between


class RecSysUser(HttpUser):
    wait_time = between(0.01, 0.2)

    @task(5)
    def predict(self):
        # 16 features total (8 user + 8 item) matching training script default
        features = [random.random() for _ in range(16)]
        self.client.post("/predict", json={"features": features})

    @task(1)
    def predict_batch(self):
        batch = [[random.random() for _ in range(16)] for _ in range(8)]
        self.client.post("/predict/batch", json={"features": batch})
