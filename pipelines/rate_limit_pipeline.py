import os
from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import time


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        requests_per_minute: Optional[int] = None
        requests_per_hour: Optional[int] = None
        sliding_window_limit: Optional[int] = None
        sliding_window_minutes: Optional[int] = None

    def __init__(self):
        self.type = "filter"
        self.name = "Rate Limit Filter"

        self.valves = self.Valves(
            **{
                "pipelines": os.getenv("RATE_LIMIT_PIPELINES", "*").split(","),
                "requests_per_minute": int(
                    os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", 10)
                ),
                "requests_per_hour": int(
                    os.getenv("RATE_LIMIT_REQUESTS_PER_HOUR", 1000)
                ),
                "sliding_window_limit": int(
                    os.getenv("RATE_LIMIT_SLIDING_WINDOW_LIMIT", 100)
                ),
                "sliding_window_minutes": int(
                    os.getenv("RATE_LIMIT_SLIDING_WINDOW_MINUTES", 15)
                ),
            }
        )

        self.user_requests = {}

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    def prune_requests(self, user_id: str):
        now = time.time()
        if user_id in self.user_requests:
            self.user_requests[user_id] = [
                req
                for req in self.user_requests[user_id]
                if (
                    (self.valves.requests_per_minute is not None and now - req < 60)
                    or (self.valves.requests_per_hour is not None and now - req < 3600)
                    or (
                        self.valves.sliding_window_limit is not None
                        and now - req < self.valves.sliding_window_minutes * 60
                    )
                )
            ]

    def log_request(self, user_id: str):
        now = time.time()
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        self.user_requests[user_id].append(now)

    def rate_limited(self, user_id: str) -> tuple[bool, str]:
        self.prune_requests(user_id)

        user_reqs = self.user_requests.get(user_id, [])

        if self.valves.requests_per_minute is not None:
            requests_last_minute = sum(1 for req in user_reqs if time.time() - req < 60)
            if requests_last_minute >= self.valves.requests_per_minute:
                return True, f"{requests_last_minute}/{self.valves.requests_per_minute} requests per minute limit reached."

        if self.valves.requests_per_hour is not None:
            requests_last_hour = sum(1 for req in user_reqs if time.time() - req < 3600)
            if requests_last_hour >= self.valves.requests_per_hour:
                return True, f"{requests_last_hour}/{self.valves.requests_per_hour} requests per hour limit reached."

        if self.valves.sliding_window_limit is not None:
            requests_in_window = len(user_reqs)
            if requests_in_window >= self.valves.sliding_window_limit:
                return True, f"{requests_in_window}/{self.valves.sliding_window_limit} requests in {self.valves.sliding_window_minutes} minutes limit reached."

        return False, ""

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")
        print(body)
        print(user)

        if user.get("role", "admin") == "user":
            user_id = user["id"] if user and "id" in user else "default_user"
            is_limited, alert_message = self.rate_limited(user_id)
            
            if is_limited:
                raise Exception(f"Rate limit exceeded: {alert_message} Please try again later.")

            self.log_request(user_id)
        return body