#!/usr/bin/env python3
"""Simulates a real AI discovery query — "who would ChatGPT recommend if a
customer asks for a [business type] in [location]?" — and checks whether
THIS business shows up, and roughly where.

This answers the product's core promise ("is AI finding your business — or
your competitors?") the way it actually happens: a real AI answer engine
searching the live web for a generic "best plumber near me"-style query,
not a description of the business's own page.

Uses OpenAI (ChatGPT) with its web_search tool, in two steps:
  1. responses.create + web_search  → ChatGPT's real, web-grounded answer
  2. responses.parse                → extract the ranked businesses as data
The "did we appear / at what position" match is then done in Python (domain
+ name matching), which is more reliable than asking the model to grade
itself.

Requires an OpenAI API key in OPENAI_API_KEY — see is_available().
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

from openai import OpenAI
from pydantic import BaseModel

# ChatGPT model that runs the live web search. Swap if your account exposes
# a different tier. The extraction step can use a cheaper model since it's
# just structuring text the first call already produced.
SEARCH_MODEL = "gpt-4.1"
EXTRACT_MODEL = "gpt-4.1-mini"


class RecommendedBusiness(BaseModel):
    name: str
    url: str | None = None


class RankedBusinesses(BaseModel):
    businesses: list[RecommendedBusiness]


class DiscoveryResult:
    """Plain result object the app renders — not a pydantic model because it
    carries computed fields (appears/position) alongside the raw answer."""

    def __init__(self, query, answer_text, recommended, appears, position, target_label):
        self.query = query
        self.answer_text = answer_text
        self.recommended = recommended  # list[RecommendedBusiness], ranked
        self.appears = appears          # bool
        self.position = position        # int | None (1-indexed) when appears
        self.target_label = target_label


def is_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _domain(url: str | None) -> str:
    if not url:
        return ""
    if "://" not in url:
        url = "https://" + url
    return urlparse(url).netloc.lower().replace("www.", "")


def derive_business_type(report) -> str:
    """Best-effort guess at the business category for the query — editable by
    the user in the UI, so it only needs to be a sensible default."""
    # Schema types that describe page structure or sub-objects, NOT the
    # business category — must be ignored (e.g. an FAQ page's Question/Answer
    # types would otherwise yield the nonsense query "best answer in ...").
    non_category = {
        "LocalBusiness", "Organization", "WebSite", "WebPage", "PostalAddress",
        "GeoCoordinates", "OpeningHoursSpecification", "BreadcrumbList", "ListItem",
        "FAQPage", "Question", "Answer", "Review", "AggregateRating", "Rating",
        "ImageObject", "SearchAction", "ContactPoint", "Place", "Person",
        "Product", "Offer", "Service", "Article", "BlogPosting", "SiteNavigationElement",
    }
    # Only the HOMEPAGE's schema is a reliable signal for the whole business —
    # inner pages carry page-specific types (FAQ, reviews, etc.).
    hp = report.pages[0] if report.pages else None
    for t in (hp.schema_types if hp else []):
        if t not in non_category and t.isalpha():
            # Split CamelCase (e.g. "RoofingContractor" -> "Roofing Contractor")
            words = re.sub(r"(?<!^)(?=[A-Z])", " ", t)
            return words.lower()
    # Fall back to a keyword in the homepage title.
    title = (hp.title if hp else "") or ""
    for kw in ("plumb", "hvac", "electric", "roof", "dentist", "dental",
               "landscap", "clean", "paint", "lawyer", "attorney", "contractor"):
        if kw in title.lower():
            return {"plumb": "plumber", "electric": "electrician",
                    "roof": "roofer", "dental": "dentist",
                    "landscap": "landscaper", "clean": "cleaning service",
                    "paint": "painter", "lawyer": "lawyer",
                    "attorney": "attorney"}.get(kw, kw)
    return "local business"


def derive_location(report) -> str:
    """Best-effort 'City, ST' (and ZIP if present) from the pages we scanned —
    editable by the user, so a rough guess is fine."""
    hp = report.pages[0] if report.pages else None
    haystacks = []
    if hp:
        haystacks = [hp.title or "", hp.meta_description or "", hp.visible_text_excerpt or ""]
    text = "  ".join(haystacks)
    # "City, ST 12345" or "City, ST"
    m = re.search(r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?),\s*([A-Z]{2})(?:\s+(\d{5}))?", text)
    if m:
        city, state, zip_ = m.group(1), m.group(2), m.group(3)
        return f"{city}, {state}" + (f" {zip_}" if zip_ else "")
    return ""


# Generic category words that are NOT distinctive to one business — used to
# avoid matching "any plumber" as if it were the target. The brand token
# (e.g. "savior" in saviorplumbing.com) is what actually identifies them.
_GENERIC_TOKENS = {
    "plumbing", "plumber", "plumbers", "hvac", "heating", "cooling", "air",
    "electric", "electrical", "electrician", "roofing", "roofer", "dental",
    "dentist", "dentistry", "landscaping", "landscaper", "cleaning", "cleaners",
    "painting", "painter", "contractor", "contractors", "services", "service",
    "repair", "the", "and", "best", "local", "co", "inc", "llc", "company",
    "group", "solutions", "pros", "experts", "home",
}


def _brand_token(report) -> str:
    """The distinctive part of the domain label — e.g. 'saviorplumbing.com'
    -> 'savior'. This is what reliably identifies THIS business vs. any other
    business in the same category."""
    label = _domain(report.normalized_url).split(".")[0]
    for g in sorted(_GENERIC_TOKENS, key=len, reverse=True):
        label = label.replace(g, "")
    return label if len(label) >= 3 else _domain(report.normalized_url).split(".")[0]


def derive_business_name(report) -> str:
    hp = report.pages[0] if report.pages else None
    title = (hp.title if hp else "") or ""
    parts = [s.strip() for s in re.split(r"[|\-–—·]", title) if s.strip()]
    # A segment naming a legal entity (Inc/LLC/Co) is almost always the brand.
    for p in parts:
        if re.search(r"\b(inc|llc|co|company)\b", p, re.I):
            return p
    # Otherwise the segment that actually contains the brand token.
    brand = _brand_token(report).lower()
    for p in parts:
        if brand and brand in re.sub(r"[^a-z0-9]", "", p.lower()):
            return p
    # Otherwise the first non-generic segment.
    for p in parts:
        toks = {t for t in re.findall(r"[a-z0-9]+", p.lower())}
        if toks - _GENERIC_TOKENS:
            return p
    return parts[0] if parts else _domain(report.normalized_url)


def build_query(business_type: str, location: str) -> str:
    loc = location.strip() or "my area"
    return f"I need a {business_type.strip()} in {loc}. Who are the best options you'd recommend, and why?"


def run_discovery(report, business_type: str, location: str) -> DiscoveryResult:
    """Run the live ChatGPT web-search query and compute whether/where this
    business appears. Raises openai's exception classes on failure — the
    caller decides how to surface them."""
    client = OpenAI()
    query = build_query(business_type, location)

    # Step 1 — real web-grounded answer from ChatGPT.
    answer_resp = client.responses.create(
        model=SEARCH_MODEL,
        tools=[{"type": "web_search"}],
        input=query,
    )
    answer_text = answer_resp.output_text or ""

    # Step 2 — structure the businesses ChatGPT named, in the order given.
    parsed = client.responses.parse(
        model=EXTRACT_MODEL,
        input=[
            {"role": "system", "content": (
                "Extract the businesses recommended in the assistant's answer, in the same order "
                "they are presented (most-recommended first). Include each business's website URL "
                "only if the answer states one. Do not invent businesses or URLs."
            )},
            {"role": "user", "content": f"Query: {query}\n\nAssistant's answer:\n{answer_text}"},
        ],
        text_format=RankedBusinesses,
    )
    recommended = parsed.output_parsed.businesses if parsed.output_parsed else []

    # Detection in Python. Three independent signals, because a business's
    # domain, its display name, and how ChatGPT refers to it often diverge —
    # e.g. "Premier Plumbing Solutions 925" lives at pps925.com, so a domain-
    # only match would miss it entirely:
    #   1. exact domain match (strongest),
    #   2. the domain brand token appears in the name/url ("savior"),
    #   3. the name's distinctive token appears AND ≥2 name tokens overlap
    #      (so "Premier Plumbing Solutions" matches, but a different
    #      "Premier Rooter" — sharing only "premier" — does not).
    target_domain = _domain(report.normalized_url)
    brand = _brand_token(report).lower()
    our_name_tokens = {t for t in re.findall(r"[a-z0-9]+", derive_business_name(report).lower()) if len(t) >= 3}
    distinctive = {t for t in our_name_tokens if t not in _GENERIC_TOKENS}

    def _matches(name: str, url: str | None) -> bool:
        biz_domain = _domain(url)
        name_l = name.lower()
        name_compact = re.sub(r"[^a-z0-9]", "", name_l)
        cand_tokens = {t for t in re.findall(r"[a-z0-9]+", name_l) if len(t) >= 3}
        if target_domain and biz_domain == target_domain:
            return True
        if brand and len(brand) >= 4 and (brand in name_compact or (biz_domain and brand in biz_domain)):
            return True
        if distinctive & cand_tokens and len(our_name_tokens & cand_tokens) >= 2:
            return True
        return False

    appears = False
    position = None
    for i, biz in enumerate(recommended, start=1):
        if _matches(biz.name or "", biz.url):
            appears = True
            position = i
            break

    # Fallback: the extractor may have dropped a business the answer text
    # names — scan the raw answer for our exact domain or our full name.
    answer_compact = re.sub(r"[^a-z0-9]", "", answer_text.lower())
    if not appears:
        our_name_compact = re.sub(r"[^a-z0-9]", "", derive_business_name(report).lower())
        if (target_domain and target_domain in answer_text.lower()) or (
            len(our_name_compact) >= 6 and our_name_compact in answer_compact
        ):
            appears = True

    return DiscoveryResult(
        query=query,
        answer_text=answer_text,
        recommended=recommended,
        appears=appears,
        position=position,
        target_label=derive_business_name(report),
    )
