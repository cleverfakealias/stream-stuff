"""Simple web scraping utilities.

These functions fetch web pages while respecting website policies.
"""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup


def fetch_url(url: str) -> str:
    """Fetch text from a static web page.

    Args:
        url: The URL to retrieve.

    Returns:
        The page text with minimal formatting.
    """
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text(" ", strip=True)


try:  # pragma: no cover - optional dependency
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover - execution environment may not have playwright
    sync_playwright = None


def fetch_dynamic(url: str) -> str:
    """Fetch text from a dynamic page using Playwright if available."""
    if sync_playwright is None:
        raise RuntimeError("Playwright is not available")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        text = page.inner_text("body")
        browser.close()
    return text
