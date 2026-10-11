"""Small, rate-limited clients for the two scholarly APIs (stdlib only)."""
from __future__ import annotations

import email.utils
import json
import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request

LOG = logging.getLogger(__name__)
FIELDS = "paperId,title,abstract,year,publicationDate,venue,publicationVenue,externalIds,authors,url,publicationTypes"
OA_FIELDS = "id,doi,display_name,publication_year,publication_date,abstract_inverted_index,authorships,primary_location,locations,type"


class APIError(RuntimeError):
    pass


class RequestLimit(RuntimeError):
    """A deliberately bounded run; pending work will resume next time."""


class Budget:
    def __init__(self, maximum=None):
        self.maximum = maximum
        self.used = 0

    def take(self):
        if self.maximum is not None and self.used >= self.maximum:
            raise RequestLimit(f"Stopped at the requested {self.maximum} HTTP requests; rerun to resume.")
        self.used += 1


class HTTP:
    def __init__(self, name, base_url, budget, headers=None, interval=1.1, retries=5, timeout=60):
        self.name, self.base_url, self.budget = name, base_url, budget
        self.headers = {"User-Agent": "reference-finder/1.0", "Accept": "application/json", **(headers or {})}
        self.interval, self.retries, self.timeout = interval, retries, timeout
        self.last_request = 0.0

    def request(self, path, params=None, body=None, allow_not_found=False):
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = dict(self.headers)
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        for attempt in range(self.retries + 1):
            self.budget.take()
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            retry_after = None
            try:
                req = urllib.request.Request(url, headers=headers, data=data)
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and allow_not_found:
                    return None
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    hint = " Check the API key and account quota." if exc.code in (401, 402, 403) else ""
                    # Never print request URLs, response bodies, or headers containing credentials.
                    raise APIError(f"{self.name}: HTTP {exc.code}.{hint}") from None
                retry_after = exc.headers.get("Retry-After")
                reason = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                reason = "network error"
            except (ValueError, UnicodeError):
                reason = "invalid JSON response"
            if attempt == self.retries:
                raise APIError(f"{self.name}: {reason} after {attempt + 1} attempts; rerun to resume.") from None
            delay = min(60, 2 ** attempt + random.random())
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    try:
                        delay = max(delay, email.utils.parsedate_to_datetime(retry_after).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
            if delay > 60:
                raise APIError(f"{self.name}: server requests a wait of {delay:.0f}s; resume later.") from None
            LOG.warning("%s: %s; retrying in %.1fs", self.name, reason, delay)
            time.sleep(delay)
        raise AssertionError("unreachable")


class OpenAlex:
    def __init__(self, http):
        self.http = http

    def sources(self, query, cursor="*"):
        return self.http.request("/sources", {"search": query, "per_page": 100, "cursor": cursor})

    def works(self, source_id, year, cursor="*"):
        return self.http.request("/works", {
            "filter": f"locations.source.id:{source_id},publication_year:{year}",
            "select": OA_FIELDS, "per_page": 100, "cursor": cursor,
        })

    def doi(self, doi):
        return self.http.request("/works/" + urllib.parse.quote("https://doi.org/" + doi, safe=""),
                                 {"select": OA_FIELDS}, allow_not_found=True)


class SemanticScholar:
    def __init__(self, http):
        self.http = http

    def works(self, venue_names, year, token=None):
        params = {"venue": ",".join(venue_names), "year": year, "fields": FIELDS}
        if token:
            params["token"] = token
        # Bulk search permits a venue-only query; a title query would lose papers.
        return self.http.request("/paper/search/bulk", params)

    def dois(self, dois):
        return self.http.request("/paper/batch", {"fields": FIELDS},
                                 {"ids": ["DOI:" + doi for doi in dois]})
