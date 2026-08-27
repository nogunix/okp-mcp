"""Section anchors read from the OKP appliance's HTML mirror.

Solr indexes documentation as plain text: ``heading_h1``/``heading_h2`` carry
the section titles but nothing carries the URL fragment each section lives at,
and the fragments cannot be derived from the titles.  Red Hat assigns them in
the AsciiDoc source, so "Kafka tuning overview" is published under
``#con-config-tuning-intro-str``; deriving slugs from heading text was measured
against a live guide and matched 1 heading in 44.

The same appliance that serves Solr also ships the rendered HTML behind its own
httpd, and there each section is a ``<section id="...">`` carrying exactly the
id docs.redhat.com uses as its fragment.  Reading the outline from there keeps
the server offline while still producing links that resolve on the public site:
sampled against the live docs, 45 of a guide's 46 anchors matched byte for byte
(the odd one out is the document wrapper, which is not a section).

Coverage over a 250-document sample of the indexed corpus: every document was
reachable, 236 expose ``<section id>``, and the remaining 14 are single-topic
pages that have no subsections to link to.  Callers fall back to the
title-only outline when this module returns nothing.
"""

import logging

from collections import OrderedDict
from html.parser import HTMLParser
from typing import NamedTuple

import httpx


logger = logging.getLogger("okp_mcp.outline")

# Headings render as <h1 class="title">, <h2 class="title">, ... inside the
# section they name. Anything deeper than h4 is a formatting heading rather
# than a linkable section.
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4"})

# Wrapper element around the whole document; it carries an id but is the page
# itself rather than a section within it, so linking to it is a no-op.
_WRAPPER_ID_PREFIX = "mimir-doc--"

# Outlines are small (a list of short string pairs) but each one costs a
# ~200KB fetch, so keep recently used documents around. Sized to hold a
# working set of guides without pinning the whole corpus.
_CACHE_SIZE = 128


class Section(NamedTuple):
    """A linkable section: the URL fragment and the heading it belongs to.

    ``level`` is the nesting depth (1 for a chapter, 2 for a section within
    it, ...), which lets callers shed the deepest levels first when an
    outline is too large to render whole.
    """

    anchor: str
    title: str
    level: int = 1


class _OutlineParser(HTMLParser):
    """Collect ``(section id, heading text)`` pairs from a rendered guide.

    Sections nest, so the parser keeps a stack and attributes a heading to the
    innermost section open when it starts.  Sections without an ``id`` still
    push onto the stack -- dropping them would misattribute their headings to
    the enclosing section, which would produce a link to the wrong place.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._open_sections: list[str | None] = []
        self._collecting: str | None = None
        self._level = 1
        self._buffer: list[str] = []
        self.sections: list[Section] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "section":
            self._open_sections.append(dict(attrs).get("id"))
            return
        if tag in _HEADING_TAGS and self._open_sections:
            anchor = self._open_sections[-1]
            # One heading per section: a nested formatting heading inside an
            # already-titled section must not overwrite the section's title.
            if anchor and not any(existing.anchor == anchor for existing in self.sections):
                self._collecting = anchor
                self._level = len(self._open_sections)
                self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "section":
            if self._open_sections:
                self._open_sections.pop()
            return
        if tag in _HEADING_TAGS and self._collecting:
            title = " ".join("".join(self._buffer).split())
            if title:
                self.sections.append(Section(self._collecting, title, self._level))
            self._collecting = None

    def handle_data(self, data: str) -> None:
        if self._collecting:
            self._buffer.append(data)


def parse_outline(html: str) -> list[Section]:
    """Extract the linkable sections from a rendered documentation page.

    Levels are normalised so the shallowest section returned is level 1,
    because the dropped document wrapper otherwise pushes every real chapter
    down a level on the pages that have one.
    """
    parser = _OutlineParser()
    parser.feed(html)
    sections = [section for section in parser.sections if not section.anchor.startswith(_WRAPPER_ID_PREFIX)]
    if not sections:
        return []

    offset = min(section.level for section in sections) - 1
    return [section._replace(level=section.level - offset) for section in sections]


def _html_path(doc_id: str) -> str:
    """Map a Solr document id onto its path in the HTML mirror.

    Solr ids are the crawled file paths, so the mapping is the identity apart
    from the ``/index.html`` suffix that ``doc_uri`` strips for display.
    """
    path = doc_id if doc_id.startswith("/") else f"/{doc_id}"
    return path if path.endswith(".html") else f"{path.removesuffix('/')}/index.html"


class OutlineFetcher:
    """Fetches and caches section outlines from the HTML mirror.

    Every failure mode -- no mirror configured, mirror unreachable, document
    absent, page carrying no sections -- resolves to an empty list.  The
    outline is a navigational extra, so it must never turn a working
    ``get_document`` call into an error.
    """

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._cache: OrderedDict[str, list[Section]] = OrderedDict()

    async def get(self, doc_id: str) -> list[Section]:
        """Return the sections of a document, or an empty list if unavailable."""
        if not self._base_url:
            return []

        if doc_id in self._cache:
            self._cache.move_to_end(doc_id)
            return self._cache[doc_id]

        sections = await self._fetch(doc_id)

        self._cache[doc_id] = sections
        self._cache.move_to_end(doc_id)
        if len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return sections

    async def _fetch(self, doc_id: str) -> list[Section]:
        url = f"{self._base_url}{_html_path(doc_id)}"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # Debug, not warning: a deployment that exposes only Solr hits this
            # on every documentation fetch and the fallback is well defined.
            logger.debug("outline fetch failed for %s: %s", url, exc)
            return []

        try:
            return parse_outline(response.text)
        except (ValueError, AssertionError) as exc:
            logger.debug("outline parse failed for %s: %s", url, exc)
            return []
