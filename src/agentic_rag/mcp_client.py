"""Minimal client to communicate with external AI services via HTTP."""
from __future__ import annotations

import requests


class MCPClient:
    """Very small HTTP client for an external AI system."""

    def __init__(self, endpoint: str, api_key: str | None = None) -> None:
        self.endpoint = endpoint
        self.api_key = api_key

    def query(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = requests.post(
            self.endpoint, json={"prompt": prompt}, headers=headers, timeout=60
        )
        resp.raise_for_status()
        return resp.json().get("output", "")
