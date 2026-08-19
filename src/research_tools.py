import hashlib
import html.parser
import ipaddress
import json
import socket
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from src.literature import dedupe_records
from src.research_state import now_iso


class ResearchToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, tool_id, operation, executor, trusted=True):
        self._tools[tool_id] = {"tool_id": tool_id, "operation": operation, "executor": executor, "trusted": trusted}

    def execute(self, tool_id, request):
        tool = self._tools.get(tool_id)
        if not tool or not tool["trusted"]:
            raise ValueError("unregistered or untrusted research tool")
        if set(request) - {"query", "url", "limit", "provider_id"}:
            raise ValueError("research tool request contains unsupported orchestration fields")
        return tool["executor"](request)


def provider_descriptor(provider_id, source_kind, supported_operations, evidence_kind, availability="AVAILABLE",
                        authentication="none", result_format="json", verification_semantics=None):
    return {"provider_id": provider_id, "source_kind": source_kind, "supported_operations": list(supported_operations),
            "evidence_kind": evidence_kind, "authentication": authentication, "availability": availability,
            "result_format": result_format, "verification_semantics": verification_semantics or [],
            "evidence_modality": {"scholarly_metadata": "literature_metadata", "dataset": "secondary_dataset_analysis"}.get(evidence_kind, evidence_kind)}


def classify_provider_search_outcome(descriptor, result=None, error=None, relevance=None):
    if descriptor.get("availability") != "AVAILABLE":
        status = "PROVIDER_UNAVAILABLE"
    elif error:
        status = "NETWORK_OR_PROVIDER_ERROR"
    elif result is None or not isinstance(result, dict):
        status = "MALFORMED_RETRIEVAL"
    elif not result.get("records"):
        status = "ZERO_RESULTS"
    elif relevance is not None and not relevance.get("usable"):
        status = "RESULTS_IRRELEVANT"
    else:
        status = "USABLE_RESULTS"
    return {"provider_id": descriptor.get("provider_id"), "status": status, "error": str(error) if error else None,
            "record_count": len((result or {}).get("records", [])), "retrieval_timestamp": now_iso()}


class MultiProviderResearchSupervisor:
    def __init__(self, providers, web_search=None, max_provider_attempts=3):
        self.providers = list(providers)
        self.web_search = web_search
        self.max_provider_attempts = max_provider_attempts

    def scholarly_search(self, raw_query, normalized_query, limit, relevance_fn):
        attempts, records, raw_results = [], [], []
        for entry in self.providers[:self.max_provider_attempts]:
            descriptor, provider = entry["descriptor"], entry.get("provider")
            if descriptor.get("availability") != "AVAILABLE":
                attempts.append(classify_provider_search_outcome(descriptor)); continue
            try:
                result = provider.search(normalized_query, limit=limit)
                relevance = relevance_fn(result.get("records", []))
                attempts.append({**classify_provider_search_outcome(descriptor, result=result, relevance=relevance), "relevance": relevance})
                raw_results.append({"provider": descriptor, "result": result})
                if relevance.get("usable"):
                    records.extend(result.get("records", []))
            except Exception as exc:
                attempts.append(classify_provider_search_outcome(descriptor, error=exc))
        web_results = None
        if not records and self.web_search:
            web_results = self.web_search(normalized_query, limit=limit)
            attempts.append({"provider_id": "web_search", "status": "WEB_CANDIDATE_SOURCES",
                             "record_count": len(web_results.get("results", [])), "retrieval_timestamp": now_iso()})
        return {"raw_query": raw_query, "normalized_query": normalized_query, "attempts": attempts,
                "provider_raw_results": raw_results, "records": dedupe_records(records), "web_candidate_sources": web_results,
                "attempt_count": len(attempts), "bounded_attempt_limit": self.max_provider_attempts + (1 if self.web_search else 0)}


class _TextExtractor(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(); self.title = ""; self.parts = []; self._title = False
    def handle_starttag(self, tag, attrs):
        if tag == "title": self._title = True
    def handle_endtag(self, tag):
        if tag == "title": self._title = False
    def handle_data(self, data):
        text = " ".join(data.split())
        if text:
            if self._title: self.title += (" " if self.title else "") + text
            self.parts.append(text)


def validate_public_http_url(url, resolver=socket.getaddrinfo):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only absolute http/https URLs are allowed")
    if parsed.username or parsed.password:
        raise ValueError("URL credentials are forbidden")
    if parsed.hostname.lower() == "localhost":
        raise ValueError("localhost/private-network URL rejected")
    try:
        literal = ipaddress.ip_address(parsed.hostname)
        addresses = [literal]
    except ValueError:
        addresses = [ipaddress.ip_address(item[4][0]) for item in resolver(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)]
    if any(address.is_private or address.is_loopback or address.is_link_local or address.is_reserved for address in addresses):
        raise ValueError("localhost/private-network URL rejected")
    return parsed.geturl()


class _BoundedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, max_redirects):
        super().__init__(); self.max_redirects = max_redirects; self.redirect_count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.redirect_count += 1
        if self.redirect_count > self.max_redirects:
            raise ValueError("redirect limit exceeded")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def safe_fetch(url, store, run_id, provider_id="generic_http", timeout=15, max_bytes=2_000_000,
               max_extracted_chars=12_000, opener=None, resolver=socket.getaddrinfo, max_redirects=3):
    validated = validate_public_http_url(url, resolver=resolver)
    request = urllib.request.Request(validated, headers={"User-Agent": "researchGPT/0.1"})
    open_call = opener or urllib.request.build_opener(_BoundedRedirectHandler(max_redirects)).open
    with open_call(request, timeout=timeout) as response:
        final_url = validate_public_http_url(response.geturl(), resolver=resolver)
        content_type = response.headers.get_content_type()
        raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError("response exceeds configured size limit")
    sha = hashlib.sha256(raw).hexdigest()
    suffix = ".html" if content_type == "text/html" else ".json" if content_type == "application/json" else ".bin"
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as handle:
        handle.write(raw); temp_path = handle.name
    artifact_path = f"sources/raw/{sha}{suffix}"
    try:
        artifact = store.put_artifact(run_id, temp_path, artifact_path, f"research_tool:{provider_id}")
    finally:
        Path(temp_path).unlink(missing_ok=True)
    normalized = None
    if content_type == "text/html":
        parser = _TextExtractor(); parser.feed(raw.decode("utf-8", errors="replace"))
        normalized = {"canonical_url": final_url, "title": parser.title, "source_provider": provider_id,
                      "relevant_extracted_text": " ".join(parser.parts)[:max_extracted_chars], "truncated": len(" ".join(parser.parts)) > max_extracted_chars}
    elif content_type == "application/json":
        normalized = {"canonical_url": final_url, "source_provider": provider_id,
                      "json": json.loads(raw.decode("utf-8"))}
    return {"url": final_url, "provider_id": provider_id, "content_type": content_type,
            "retrieval_timestamp": now_iso(), "sha256": sha, "raw_artifact": artifact, "normalized": normalized,
            "request_provenance": {"requested_url": url, "timeout": timeout, "max_bytes": max_bytes,
                                   "max_redirects": max_redirects}}
