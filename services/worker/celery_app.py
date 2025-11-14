import os
from celery import Celery

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "recsys_worker",
    broker=redis_url,
    backend=redis_url,
)

app.conf.beat_schedule = {
    # Will be populated when implementing promotion/rollback automation
}
