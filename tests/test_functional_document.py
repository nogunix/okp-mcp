"""Functional tests for get_document retrieval against live Solr.

These tests prove the two behaviors that unit tests (which mock Solr) cannot:

1. The caller's query never gates retrieval. A document fetched by ID is
   returned even when the query shares few or no terms with the document
   text, because the query only drives highlighting (``hl.q``) and not the
   main ``q`` under edismax ``mm``. This is the regression behind
   GitHub issue #377 / the "Document not found" reports.
2. The visible search-result URL round-trips back into get_document. The URL
   search_portal renders (``/index.html`` stripped) resolves to the same
   document whether or not the suffix is present.

Tests seed a real document by running a portal search first, then feed the
resulting URL back into the document-fetch helpers. This keeps them robust to
index churn: whatever the live index currently returns for a stable query is
what we round-trip.

The visible URL is passed through ``_normalize_doc_id`` before the fetch, exactly
as the ``get_document`` tool does: normalization (stripping the
``access.redhat.com`` prefix to a bare path) is a required precondition of the
fetch layer, since the corpus stores path-based IDs. ``test_visible_url_...``
also asserts that the *un-normalized* URL returns zero documents, pinning down
why that normalization step matters.

Run with::

    uv run pytest -m functional -v

Requires: OKP Solr container running (``podman-compose up -d``). Tests skip
automatically if Solr is unreachable.
"""

import httpx
import pytest

from okp_mcp.config import ServerConfig
from okp_mcp.portal import _run_portal_search
from okp_mcp.portal import PortalChunk
from okp_mcp.tools.document import _fetch_document_raw
from okp_mcp.tools.document import _fetch_document_with_query
from okp_mcp.tools.document import _normalize_doc_id
from okp_mcp.types import SolrResponse


# Gibberish tokens that match no real document, proving the query cannot gate
# retrieval: a lookup by ID must still return the document.
_UNRELATED_QUERY = "qwxyz nonexistent gibberish token zzxq"


async def _seed_document() -> tuple[str, str, str]:
    """Return (visible_url, doc_id, parent_id) for a real document, skipping if Solr is down.

    Runs a stable portal search and takes the top result. ``visible_url`` is the
    ``online_source_url`` an LLM copies from search results; ``doc_id`` is that
    URL after ``_normalize_doc_id`` (the required precondition the get_document
    tool applies before any fetch); ``parent_id`` is the underlying Solr id used
    to look up highlight snippets.
    """
    config = ServerConfig()
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            chunks, _ = await _run_portal_search(
                "How do I recreate the grub configuration file",
                client=client,
                solr_endpoint=config.solr_endpoint,
                max_results=7,
            )
        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip(f"Solr not reachable at {config.solr_url}")
            raise  # unreachable, satisfies type checker

    top: PortalChunk | None = next((c for c in chunks if c.online_source_url), None)
    if top is None:
        pytest.skip("No seed document with a URL returned by portal search")
    return top.online_source_url, _normalize_doc_id(top.online_source_url), top.parent_id or ""


def _returned_docs(response: SolrResponse) -> int:
    """Return the number of documents in a Solr response body."""
    return len(response.response.docs)


@pytest.mark.functional
async def test_query_does_not_gate_document_retrieval() -> None:
    """A document fetched by ID is returned even when the query matches nothing.

    Proves the fix for the retrieval-gating bug: the caller's query drives
    highlighting only, so a poorly-overlapping (here, deliberately unrelated)
    query cannot turn a valid document lookup into "Document not found".
    """
    _, doc_id, _ = await _seed_document()
    config = ServerConfig()

    async with httpx.AsyncClient(timeout=30.0) as client:
        with_unrelated = await _fetch_document_with_query(
            doc_id, _UNRELATED_QUERY, client, solr_endpoint=config.solr_endpoint
        )

    assert _returned_docs(with_unrelated) == 1, (
        "Document lookup was gated by the query: an unrelated query returned "
        f"zero documents for {doc_id!r}. The query must only drive highlighting."
    )


@pytest.mark.functional
async def test_highlight_query_selects_passages_without_gating() -> None:
    """A relevant query yields highlights; an unrelated query still returns the doc.

    Contrasts the two query behaviors on the same document: both retrieve the
    document (retrieval is not gated), but only the relevant query is expected
    to produce highlight snippets keyed to that document.
    """
    _, doc_id, parent_id = await _seed_document()
    config = ServerConfig()

    async with httpx.AsyncClient(timeout=30.0) as client:
        relevant = await _fetch_document_with_query(
            doc_id, "recreate grub configuration file", client, solr_endpoint=config.solr_endpoint
        )
        unrelated = await _fetch_document_with_query(
            doc_id, _UNRELATED_QUERY, client, solr_endpoint=config.solr_endpoint
        )

    assert _returned_docs(relevant) == 1, f"Relevant query failed to retrieve {doc_id!r}"
    assert _returned_docs(unrelated) == 1, f"Unrelated query failed to retrieve {doc_id!r}"

    relevant_snippets = _highlight_snippets(relevant, parent_id)
    assert relevant_snippets, (
        f"Relevant query produced no highlight snippets; hl.q is not selecting passages for {doc_id!r}."
    )


@pytest.mark.functional
async def test_visible_url_round_trips_through_raw_fetch() -> None:
    """The search-result URL resolves back to the document via the raw path.

    Proves the fix for the ID-suffix bug: the URL search_portal renders
    (with ``/index.html`` stripped) resolves to the same document through the
    no-query raw fetch, which is the fallback path when the LLM omits a query.
    """
    visible_url, doc_id, _ = await _seed_document()
    config = ServerConfig()

    async with httpx.AsyncClient(timeout=30.0) as client:
        raw = await _fetch_document_raw(doc_id, client, solr_endpoint=config.solr_endpoint)
        raw_unnormalized = await _fetch_document_raw(visible_url, client, solr_endpoint=config.solr_endpoint)

    assert _returned_docs(raw) == 1, (
        f"Raw fetch could not resolve the normalized id {doc_id!r}. The id/view_uri "
        "suffix normalization is not round-tripping."
    )
    assert _returned_docs(raw_unnormalized) == 0, (
        f"The un-normalized URL {visible_url!r} unexpectedly resolved. This test "
        "pins down why get_document must call _normalize_doc_id before fetching: "
        "the corpus stores path-based ids, so the full URL matches nothing."
    )


def _highlight_snippets(response: SolrResponse, parent_id: str) -> list[str]:
    """Return highlight snippets for the parent doc, tolerant of key form.

    Solr keys highlighting by document id. The seed's parent_id should match,
    but if the index keys differ (suffix form), fall back to any non-empty
    main_content snippets present in the response.
    """
    keyed = response.highlighting.get(parent_id, {}).get("main_content", [])
    if keyed:
        return keyed
    for hit in response.highlighting.values():
        snippets = hit.get("main_content", [])
        if snippets:
            return snippets
    return []
