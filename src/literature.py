import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from src.research_state import now_iso


class LiteratureProvider:
    provider_name = "abstract"

    def search(self, query, limit=10):
        raise NotImplementedError

    def fetch_metadata(self, identifier):
        raise NotImplementedError

    def fetch_abstract(self, identifier):
        record = self.fetch_metadata(identifier)
        return record.get("abstract")

    def verify_record(self, record):
        required = ["identifier", "title", "source_provider", "retrieval_timestamp"]
        missing = [key for key in required if not record.get(key)]
        record = dict(record)
        record["verification_status"] = "VERIFIED_METADATA" if not missing else "PARTIAL_METADATA"
        record["verification_notes"] = missing
        return record


def _abstract_from_openalex(inv):
    if not inv:
        return None
    words = sorted(inv.items(), key=lambda kv: kv[1])
    return " ".join(word for word, _ in words)


class OpenAlexProvider(LiteratureProvider):
    provider_name = "openalex"

    def __init__(self, email=None, timeout=20):
        self.email = email or os.environ.get("OPENALEX_EMAIL")
        self.timeout = timeout

    def _get_json(self, url):
        headers = {"User-Agent": "researchGPT/0.1"}
        if self.email:
            headers["mailto"] = self.email
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def search(self, query, limit=10):
        params = {"search": query, "per-page": str(limit)}
        if self.email:
            params["mailto"] = self.email
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        raw = self._get_json(url)
        records = [self._normalize(item, query, raw_provider_response=item) for item in raw.get("results", [])]
        return {"provider": self.provider_name, "query": query, "retrieval_timestamp": now_iso(), "raw_response": raw, "records": records}

    def fetch_metadata(self, identifier):
        url_id = identifier
        if identifier.startswith("https://openalex.org/"):
            url_id = identifier.rsplit("/", 1)[-1]
        raw = self._get_json(f"https://api.openalex.org/works/{urllib.parse.quote(url_id)}")
        return self._normalize(raw, None, raw_provider_response=raw)

    def _normalize(self, item, query, raw_provider_response=None):
        doi = item.get("doi")
        authors = []
        for authorship in item.get("authorships", []) or []:
            author = authorship.get("author", {})
            if author.get("display_name"):
                authors.append(author["display_name"])
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        record = {
            "identifier": item.get("id"),
            "title": item.get("title"),
            "authors": authors,
            "year": item.get("publication_year"),
            "venue": source.get("display_name"),
            "doi": doi,
            "stable_url": doi or item.get("id"),
            "abstract": _abstract_from_openalex(item.get("abstract_inverted_index")),
            "source_provider": self.provider_name,
            "retrieval_timestamp": now_iso(),
            "search_query": query,
            "full_text_available": bool((primary_location.get("landing_page_url") or primary_location.get("pdf_url"))),
            "limitations": [] if item.get("abstract_inverted_index") else ["abstract_unavailable"],
        }
        return self.verify_record(record)


class LiteratureCache:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get("RESEARCH_LITERATURE_CACHE", ".literature-cache"))
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, provider, query):
        digest = hashlib.sha256(f"{provider}\0{query}".encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def get(self, provider, query, max_age_seconds=60 * 60 * 24 * 30):
        path = self._key(provider, query)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - path.stat().st_mtime > max_age_seconds:
            return None
        return data

    def put(self, provider, query, result):
        path = self._key(provider, query)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return str(path)


class CachedLiteratureProvider(LiteratureProvider):
    def __init__(self, provider=None, cache=None):
        self.provider = provider or OpenAlexProvider()
        self.cache = cache or LiteratureCache()
        self.provider_name = self.provider.provider_name

    def search(self, query, limit=10):
        cached = self.cache.get(self.provider_name, query)
        if cached:
            cached["cache_hit"] = True
            return cached
        result = self.provider.search(query, limit=limit)
        result["cache_hit"] = False
        result["cache_path"] = self.cache.put(self.provider_name, query, result)
        return result

    def fetch_metadata(self, identifier):
        return self.provider.fetch_metadata(identifier)


def literature_provider_from_env():
    provider = os.environ.get("RESEARCH_LITERATURE_PROVIDER", "openalex").lower()
    if provider == "openalex":
        return CachedLiteratureProvider(OpenAlexProvider())
    raise ValueError(f"unknown RESEARCH_LITERATURE_PROVIDER: {provider}")


def dedupe_records(records):
    seen = set()
    unique = []
    for record in records:
        key = record.get("doi") or record.get("identifier") or (str(record.get("title", "")).lower(), record.get("year"))
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique
