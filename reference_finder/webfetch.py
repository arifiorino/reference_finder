"""Polite, cached retrieval of public bibliography pages and the DBLP API."""
from __future__ import annotations

import hashlib
import email.utils
import json
import logging
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from .api import Budget, APIError

LOG = logging.getLogger(__name__)
USER_AGENT = "reference-finder/2.0 (academic bibliography; Python)"


class FetchError(APIError):
    pass


class WebFetcher:
    def __init__(self, cache_dir="data/page_cache", refresh=False, interval=1.0, budget=None, timeout=45):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh, self.interval, self.timeout = refresh, interval, timeout
        self.budget = budget or Budget()
        self.last = {}
        self.session = {}

    def _validate(self, body, host):
        if not isinstance(body, str) or not body.strip():
            raise FetchError(f"{host}: empty or invalid response")
        if any(token in body.lower() for token in ("anubis_version", "anubis_challenge", "making sure you're not a bot", "just a moment...", "cf-chl-")):
            raise FetchError(f"{host}: bot challenge; page was not parsed or cached.")
        if host == "sparql.dblp.org":
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict) or not isinstance(payload.get("results", {}).get("bindings"), list):
                    raise ValueError("Missing bindings")
            except (ValueError, AttributeError, TypeError):
                raise FetchError(f"{host}: invalid SPARQL response; not cached") from None

    def __call__(self, url):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https" or parts.hostname not in {
            "www.usenix.org", "usenix.org", "www.iacr.org", "iacr.org", "sparql.dblp.org", "www.sigsac.org", "sigsac.org"
        }:
            raise FetchError("Unsupported source URL; use a documented public source or saved HTML.")
        if url in self.session:
            return self.session[url]
        path = self.cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if path.exists() and not self.refresh:
            try:
                entry = json.loads(path.read_text())
                if isinstance(entry, dict) and entry.get("url") == url:
                    self._validate(entry.get("body"), parts.hostname)
                    self.session[url] = entry["body"]
                    return entry["body"]
            except (ValueError, KeyError, FetchError):
                pass
        for attempt in range(3):
            self.budget.take()
            time.sleep(max(0, self.interval - (time.monotonic() - self.last.get(parts.hostname, 0))))
            self.last[parts.hostname] = time.monotonic()
            headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json" if parts.hostname == "sparql.dblp.org" else "text/html"}
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=self.timeout) as r:
                    body = r.read().decode(r.headers.get_content_charset() or "utf-8", errors="replace")
                self._validate(body, parts.hostname)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"url": url, "fetched_at": time.time(), "body": body}, ensure_ascii=False))
                temporary.replace(path)
                self.session[url] = body
                return body
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise FetchError(f"{parts.hostname}: HTTP {exc.code} ({parts.path})") from None
                try:
                    delay = max(2 ** (attempt + 1), float(exc.headers.get("Retry-After", 0)))
                except ValueError:
                    delay = 2 ** (attempt + 1)
                    try:
                        delay = max(delay, email.utils.parsedate_to_datetime(exc.headers["Retry-After"]).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
                if delay > 60 or attempt == 2:
                    raise FetchError(f"{parts.hostname}: HTTP {exc.code}; resume later") from None
                LOG.warning("%s: HTTP %s, waiting %.0fs", parts.hostname, exc.code, delay)
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == 2:
                    raise FetchError(f"{parts.hostname}: network request failed; resume later") from None
                time.sleep(2 ** (attempt + 1))
