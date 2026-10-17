"""Polite HTTP client: per-host rate limit, backoff on 429 and 5xx, response cache."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass
class Response:
    url: str
    status: int
    text: str
    from_cache: bool
    elapsed_s: float


class ResponseCache:
    """SQLite cache keyed by (namespace, url)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, url TEXT, status INT,"
            " body TEXT, fetched_at REAL)"
        )
        self._lock = threading.Lock()

    @staticmethod
    def key(namespace: str, url: str) -> str:
        return hashlib.sha256(f"{namespace}|{url}".encode()).hexdigest()

    def get(self, namespace: str, url: str) -> tuple[int, str] | None:
        with self._lock:
            row = self._con.execute(
                "SELECT status, body FROM cache WHERE key=?", (self.key(namespace, url),)
            ).fetchone()
        return (int(row[0]), str(row[1])) if row else None

    def put(self, namespace: str, url: str, status: int, body: str) -> None:
        with self._lock:
            self._con.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)",
                (self.key(namespace, url), url, status, body, time.time()),
            )
            self._con.commit()


class RateLimiter:
    """Minimum interval between requests to one host, stretched when responses slow down."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = min_interval_s
        self._last: dict[str, float] = {}
        self._penalty: dict[str, float] = {}

    def interval(self, host: str) -> float:
        return self.min_interval_s + self._penalty.get(host, 0.0)

    def wait(self, host: str, now: float | None = None, sleep: bool = True) -> float:
        """Block until the host is due; returns the delay applied."""
        now = time.monotonic() if now is None else now
        last = self._last.get(host)
        delay = 0.0 if last is None else max(0.0, last + self.interval(host) - now)
        if sleep and delay > 0:
            time.sleep(delay)
        self._last[host] = now + delay
        return delay

    def observe(self, host: str, elapsed_s: float) -> None:
        """Responses slower than 2 s add twice the excess as extra spacing (capped at 30 s)."""
        self._penalty[host] = min(30.0, max(0.0, elapsed_s - 2.0) * 2)


def backoff_delay(attempt: int, retry_after: str | None = None, base: float = 2.0) -> float:
    """Exponential backoff, honouring a numeric Retry-After header; capped at 300 s."""
    if retry_after and retry_after.isdigit():
        return min(300.0, float(retry_after))
    return float(min(300.0, base * (2**attempt)))


class PoliteClient:
    def __init__(
        self,
        user_agent: str,
        cache_path: Path,
        min_interval_s: float = 1.0,
        max_retries: int = 5,
        timeout_s: float = 60.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.cache = ResponseCache(cache_path)
        self.limiter = RateLimiter(min_interval_s)
        self.max_retries = max_retries
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, **(headers or {})},
            timeout=timeout_s,
            follow_redirects=True,
        )

    def get(self, url: str, namespace: str = "default", use_cache: bool = True) -> Response:
        if use_cache:
            hit = self.cache.get(namespace, url)
            if hit is not None:
                return Response(url, hit[0], hit[1], True, 0.0)
        host = urlparse(url).netloc
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait(host)
            t0 = time.monotonic()
            try:
                r = self._http.get(url)
            except httpx.HTTPError as exc:
                last_exc = exc
                time.sleep(backoff_delay(attempt))
                continue
            elapsed = time.monotonic() - t0
            self.limiter.observe(host, elapsed)
            if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                time.sleep(backoff_delay(attempt, r.headers.get("Retry-After")))
                continue
            if r.status_code < 500:
                self.cache.put(namespace, url, r.status_code, r.text)
            return Response(url, r.status_code, r.text, False, elapsed)
        raise RuntimeError(f"giving up on {url}: {last_exc}")

    def close(self) -> None:
        self._http.close()
