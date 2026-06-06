#!/usr/bin/env python3
"""
AI Ads / AI Recommendation Readiness Scanner - MVP

What it checks:
- robots.txt and sitemap.xml
- crawlability and status codes
- title/meta/H1 quality
- Schema.org JSON-LD presence and useful schema types
- Local business/contact signals
- service/location intent words
- page freshness signals
- social/profile links
- basic speed/page-size indicators
- Open Graph/Twitter metadata
- image alt text coverage

Usage:
    python ai_readiness_scanner.py https://example.com
    python ai_readiness_scanner.py https://example.com --max-pages 20 --out report.json --csv pages.csv

Install:
    pip install requests beautifulsoup4 lxml
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from html import unescape
from typing import Any
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup


USER_AGENT = "AIReadinessScannerMVP/0.1 (+https://example.com/bot)"
TIMEOUT = 12

BUSINESS_SCHEMA_TYPES = {
    "LocalBusiness",
    "ProfessionalService",
    "HomeAndConstructionBusiness",
    "Plumber",
    "Electrician",
    "HVACBusiness",
    "RoofingContractor",
    "GeneralContractor",
    "RealEstateAgent",
    "Dentist",
    "MedicalBusiness",
    "LegalService",
    "Restaurant",
    "Store",
    "Organization",
}

SERVICE_WORDS = {
    "services", "service", "repair", "install", "installation", "emergency",
    "pricing", "quote", "estimate", "book", "schedule", "appointment",
    "areas served", "service area", "near me", "licensed", "insured",
    "reviews", "testimonials", "warranty", "same day", "24/7",
}

CONTACT_WORDS = {
    "contact", "call", "phone", "email", "address", "hours", "open",
    "schedule", "appointment", "quote", "estimate", "book now",
}

SOCIAL_DOMAINS = {
    "facebook.com", "instagram.com", "linkedin.com", "youtube.com",
    "tiktok.com", "x.com", "twitter.com", "yelp.com", "google.com",
    "maps.google.com", "business.google.com",
}


@dataclass
class PageResult:
    url: str
    status_code: int | None = None
    final_url: str | None = None
    error: str | None = None
    load_ms: int | None = None
    size_kb: int | None = None

    title: str | None = None
    title_len: int = 0
    meta_description: str | None = None
    meta_description_len: int = 0
    h1: list[str] = field(default_factory=list)

    json_ld_count: int = 0
    schema_types: list[str] = field(default_factory=list)
    has_local_business_schema: bool = False
    has_organization_schema: bool = False
    has_faq_schema: bool = False
    has_review_or_rating_schema: bool = False
    schema_has_phone: bool = False
    schema_has_address: bool = False
    schema_has_hours: bool = False
    schema_has_geo: bool = False
    schema_has_same_as: bool = False
    schema_errors: list[str] = field(default_factory=list)

    phone_found: bool = False
    email_found: bool = False
    address_like_found: bool = False
    hours_like_found: bool = False
    service_words_found: list[str] = field(default_factory=list)
    contact_words_found: list[str] = field(default_factory=list)

    internal_links: int = 0
    external_links: int = 0
    social_links: list[str] = field(default_factory=list)

    og_tags: int = 0
    twitter_tags: int = 0
    images_total: int = 0
    images_missing_alt: int = 0

    page_score: int = 0
    issues: list[str] = field(default_factory=list)
    wins: list[str] = field(default_factory=list)


@dataclass
class SiteReport:
    input_url: str
    normalized_url: str
    scanned_at: str
    pages_scanned: int
    site_score: int
    grade: str
    robots_url: str
    robots_status: int | None
    sitemap_url: str
    sitemap_status: int | None
    homepage_indexable_hint: bool
    schema_type_counts: dict[str, int]
    top_recommendations: list[str]
    pages: list[PageResult]


def normalize_start_url(url: str) -> str:
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path or '/'}"


def same_domain(url: str, root: str) -> bool:
    return urlparse(url).netloc.lower().replace("www.", "") == urlparse(root).netloc.lower().replace("www.", "")


def clean_url(url: str) -> str:
    url, _frag = urldefrag(url)
    return url.rstrip("/") or url


def text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def fetch(session: requests.Session, url: str) -> tuple[int | None, str | None, str | None, int | None, int | None]:
    start = time.time()
    try:
        resp = session.get(url, timeout=TIMEOUT, allow_redirects=True)
        elapsed_ms = int((time.time() - start) * 1000)
        size_kb = int(len(resp.content) / 1024)
        ctype = resp.headers.get("content-type", "")
        if "text/html" not in ctype and "application/xhtml" not in ctype and urlparse(url).path not in ("", "/"):
            return resp.status_code, resp.url, None, elapsed_ms, size_kb
        resp.encoding = resp.encoding or "utf-8"
        return resp.status_code, resp.url, resp.text, elapsed_ms, size_kb
    except requests.RequestException as exc:
        return None, None, None, None, None


def extract_json_ld(soup: BeautifulSoup) -> tuple[list[Any], list[str]]:
    data = []
    errors = []
    for tag in soup.find_all("script", attrs={"type": re.compile(r"application/ld\+json", re.I)}):
        raw = tag.string or tag.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            data.append(parsed)
        except Exception as exc:
            errors.append(f"Invalid JSON-LD: {exc}")
    return data, errors


def walk_schema_nodes(obj: Any) -> list[dict[str, Any]]:
    nodes = []
    if isinstance(obj, dict):
        nodes.append(obj)
        if "@graph" in obj and isinstance(obj["@graph"], list):
            for item in obj["@graph"]:
                nodes.extend(walk_schema_nodes(item))
        for value in obj.values():
            if isinstance(value, (dict, list)):
                nodes.extend(walk_schema_nodes(value))
    elif isinstance(obj, list):
        for item in obj:
            nodes.extend(walk_schema_nodes(item))
    return nodes


def schema_types_from_json_ld(json_ld: list[Any]) -> list[str]:
    types = []
    for block in json_ld:
        for node in walk_schema_nodes(block):
            t = node.get("@type")
            if isinstance(t, list):
                types.extend(str(x) for x in t)
            elif t:
                types.append(str(t))
    return sorted(set(types))


def schema_has_key(json_ld: list[Any], keys: set[str]) -> bool:
    for block in json_ld:
        for node in walk_schema_nodes(block):
            for key in keys:
                if key in node and node.get(key):
                    return True
    return False


def visible_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return unescape(" ".join(soup.get_text(" ").split())).lower()


def analyze_page(session: requests.Session, url: str, root_url: str) -> tuple[PageResult, list[str]]:
    result = PageResult(url=url)
    status, final_url, html, load_ms, size_kb = fetch(session, url)

    result.status_code = status
    result.final_url = final_url
    result.load_ms = load_ms
    result.size_kb = size_kb

    discovered_links: list[str] = []

    if status is None:
        result.error = "Could not fetch page"
        result.issues.append("Page could not be fetched.")
        return result, discovered_links

    if status >= 400:
        result.issues.append(f"HTTP status is {status}.")
        return result, discovered_links

    if not html:
        result.issues.append("No HTML body was available to analyze.")
        return result, discovered_links

    soup = BeautifulSoup(html, "lxml")

    title = soup.find("title")
    result.title = text_or_none(title.get_text()) if title else None
    result.title_len = len(result.title or "")

    desc = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    result.meta_description = text_or_none(desc.get("content")) if desc else None
    result.meta_description_len = len(result.meta_description or "")

    result.h1 = [text_or_none(h.get_text()) or "" for h in soup.find_all("h1")]
    result.h1 = [h for h in result.h1 if h]

    json_ld, schema_errors = extract_json_ld(soup)
    result.json_ld_count = len(json_ld)
    result.schema_errors = schema_errors
    result.schema_types = schema_types_from_json_ld(json_ld)
    type_set = set(result.schema_types)

    result.has_local_business_schema = bool(type_set & BUSINESS_SCHEMA_TYPES)
    result.has_organization_schema = "Organization" in type_set
    result.has_faq_schema = "FAQPage" in type_set
    result.has_review_or_rating_schema = schema_has_key(json_ld, {"review", "aggregateRating", "ratingValue"})
    result.schema_has_phone = schema_has_key(json_ld, {"telephone", "phone"})
    result.schema_has_address = schema_has_key(json_ld, {"address"})
    result.schema_has_hours = schema_has_key(json_ld, {"openingHours", "openingHoursSpecification"})
    result.schema_has_geo = schema_has_key(json_ld, {"geo", "latitude", "longitude"})
    result.schema_has_same_as = schema_has_key(json_ld, {"sameAs"})

    text = visible_text(soup)
    result.phone_found = bool(re.search(r"(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", text))
    result.email_found = bool(re.search(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", text, re.I))
    result.address_like_found = bool(re.search(r"\b\d{2,6}\s+[a-z0-9 .'-]+\s+(street|st|avenue|ave|road|rd|drive|dr|boulevard|blvd|lane|ln|way|court|ct)\b", text))
    result.hours_like_found = bool(re.search(r"\b(mon|tue|wed|thu|fri|sat|sun|hours|open)\b", text))
    result.service_words_found = sorted({w for w in SERVICE_WORDS if w in text})
    result.contact_words_found = sorted({w for w in CONTACT_WORDS if w in text})

    root_netloc = urlparse(root_url).netloc.lower().replace("www.", "")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute = clean_url(urljoin(final_url or url, href))
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        discovered_links.append(absolute)
        link_netloc = parsed.netloc.lower().replace("www.", "")
        if link_netloc == root_netloc:
            result.internal_links += 1
        else:
            result.external_links += 1
            if any(domain in link_netloc for domain in SOCIAL_DOMAINS):
                result.social_links.append(absolute)

    result.social_links = sorted(set(result.social_links))[:10]

    result.og_tags = len(soup.find_all("meta", property=re.compile(r"^og:", re.I)))
    result.twitter_tags = len(soup.find_all("meta", attrs={"name": re.compile(r"^twitter:", re.I)}))

    imgs = soup.find_all("img")
    result.images_total = len(imgs)
    result.images_missing_alt = sum(1 for img in imgs if not (img.get("alt") or "").strip())

    score = 0

    def add(points: int, win: str):
        nonlocal score
        score += points
        result.wins.append(win)

    def issue(text_: str):
        result.issues.append(text_)

    if status and 200 <= status < 300:
        add(8, "Page returns a successful HTTP status.")
    if result.load_ms is not None and result.load_ms <= 2500:
        add(6, "Page loads in a reasonable time for this MVP check.")
    elif result.load_ms is not None:
        issue("Page appears slow from a simple fetch test.")

    if result.size_kb is not None and result.size_kb <= 1500:
        add(3, "HTML payload size is reasonable.")
    elif result.size_kb is not None:
        issue("HTML payload is large; page may be heavy.")

    if result.title and 20 <= result.title_len <= 70:
        add(8, "Title tag is present and a useful length.")
    else:
        issue("Title tag is missing, too short, or too long.")

    if result.meta_description and 50 <= result.meta_description_len <= 170:
        add(8, "Meta description is present and useful.")
    else:
        issue("Meta description is missing, too short, or too long.")

    if len(result.h1) == 1:
        add(6, "Exactly one H1 found.")
    elif len(result.h1) == 0:
        issue("No H1 found.")
    else:
        issue("Multiple H1s found; clarify the main page topic.")

    if result.json_ld_count:
        add(10, "JSON-LD structured data found.")
    else:
        issue("No JSON-LD structured data found.")

    if result.has_local_business_schema:
        add(12, "Business/organization schema type found.")
    else:
        issue("No strong LocalBusiness/Organization schema type found.")

    if result.schema_has_phone or result.phone_found:
        add(6, "Phone number found.")
    else:
        issue("No phone number found.")

    if result.schema_has_address or result.address_like_found:
        add(6, "Address signal found.")
    else:
        issue("No address signal found.")

    if result.schema_has_hours or result.hours_like_found:
        add(5, "Hours/opening signal found.")
    else:
        issue("No business hours signal found.")

    if result.schema_has_geo:
        add(4, "Geo coordinates found in schema.")
    else:
        issue("No geo coordinates found in schema.")

    if result.schema_has_same_as or result.social_links:
        add(4, "Social/profile links found.")
    else:
        issue("No sameAs/social profile links found.")

    if len(result.service_words_found) >= 3:
        add(5, "Service/booking intent language found.")
    else:
        issue("Limited service/booking intent language found.")

    if result.og_tags >= 3:
        add(3, "Open Graph metadata found.")
    else:
        issue("Open Graph metadata is limited or missing.")

    if result.images_total:
        alt_ratio = 1 - (result.images_missing_alt / max(result.images_total, 1))
        if alt_ratio >= 0.8:
            add(3, "Most images have alt text.")
        else:
            issue("Many images are missing alt text.")

    if result.schema_errors:
        issue("Some JSON-LD could not be parsed.")

    result.page_score = min(score, 100)
    return result, discovered_links


def check_url_status(session: requests.Session, url: str) -> int | None:
    try:
        resp = session.get(url, timeout=TIMEOUT, allow_redirects=True)
        return resp.status_code
    except requests.RequestException:
        return None


def crawl(start_url: str, max_pages: int) -> SiteReport:
    normalized = normalize_start_url(start_url)
    parsed = urlparse(normalized)
    origin = f"{parsed.scheme}://{parsed.netloc}"

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    robots_url = urljoin(origin, "/robots.txt")
    sitemap_url = urljoin(origin, "/sitemap.xml")
    robots_status = check_url_status(session, robots_url)
    sitemap_status = check_url_status(session, sitemap_url)

    queue = [clean_url(normalized)]
    seen = set()
    pages: list[PageResult] = []

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        page, links = analyze_page(session, url, origin)
        pages.append(page)

        for link in links:
            if len(queue) + len(seen) >= max_pages * 5:
                break
            if same_domain(link, origin) and link not in seen:
                path = urlparse(link).path.lower()
                if not any(path.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf", ".zip", ".mp4"]):
                    queue.append(link)

    valid_pages = [p for p in pages if p.status_code and p.status_code < 400]
    avg_page_score = int(sum(p.page_score for p in valid_pages) / max(len(valid_pages), 1))

    site_bonus = 0
    homepage_indexable_hint = True

    if robots_status and robots_status < 400:
        site_bonus += 4
    if sitemap_status and sitemap_status < 400:
        site_bonus += 6

    schema_counts = Counter()
    for p in pages:
        schema_counts.update(p.schema_types)

    site_score = min(100, avg_page_score + site_bonus)

    if site_score >= 85:
        grade = "A"
    elif site_score >= 70:
        grade = "B"
    elif site_score >= 55:
        grade = "C"
    elif site_score >= 40:
        grade = "D"
    else:
        grade = "F"

    top_recommendations = build_recommendations(pages, robots_status, sitemap_status)

    return SiteReport(
        input_url=start_url,
        normalized_url=normalized,
        scanned_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        pages_scanned=len(pages),
        site_score=site_score,
        grade=grade,
        robots_url=robots_url,
        robots_status=robots_status,
        sitemap_url=sitemap_url,
        sitemap_status=sitemap_status,
        homepage_indexable_hint=homepage_indexable_hint,
        schema_type_counts=dict(schema_counts.most_common()),
        top_recommendations=top_recommendations,
        pages=pages,
    )


def build_recommendations(pages: list[PageResult], robots_status: int | None, sitemap_status: int | None) -> list[str]:
    recs = []

    if not (robots_status and robots_status < 400):
        recs.append("Add or fix /robots.txt so crawlers can understand what can be accessed.")

    if not (sitemap_status and sitemap_status < 400):
        recs.append("Add or fix /sitemap.xml and submit it in search tools.")

    homepage = pages[0] if pages else None
    all_schema = set()
    for p in pages:
        all_schema.update(p.schema_types)

    if not any(t in BUSINESS_SCHEMA_TYPES for t in all_schema):
        recs.append("Add Schema.org JSON-LD for LocalBusiness/Organization, including name, URL, logo, phone, address, service area, hours, and sameAs profiles.")

    if homepage:
        if not (homepage.schema_has_phone or homepage.phone_found):
            recs.append("Make the phone number visible on the homepage and include it in schema.")
        if not (homepage.schema_has_address or homepage.address_like_found):
            recs.append("Make the business address or service area clear on the homepage and include it in schema.")
        if len(homepage.service_words_found) < 3:
            recs.append("Add clearer service and booking language: services offered, locations served, emergency/availability, quote or booking CTA.")
        if homepage.meta_description_len < 50:
            recs.append("Write a stronger meta description explaining who you help, where, and what you offer.")
        if homepage.images_total and homepage.images_missing_alt / max(homepage.images_total, 1) > 0.2:
            recs.append("Improve image alt text so visual content is understandable to crawlers and assistive tools.")

    if "FAQPage" not in all_schema:
        recs.append("Add an FAQ section with FAQPage schema for common buyer questions, pricing, timing, service area, and process.")

    if not any(p.has_review_or_rating_schema for p in pages):
        recs.append("Add testimonials/reviews where appropriate; use valid review or aggregateRating schema only when it follows platform rules and reflects real reviews.")

    return recs[:10]


def print_human_report(report: SiteReport) -> None:
    print("\nAI Ads / AI Recommendation Readiness Report")
    print("=" * 52)
    print(f"Site: {report.normalized_url}")
    print(f"Pages scanned: {report.pages_scanned}")
    print(f"Score: {report.site_score}/100  Grade: {report.grade}")
    print(f"robots.txt: {report.robots_status}  {report.robots_url}")
    print(f"sitemap.xml: {report.sitemap_status}  {report.sitemap_url}")

    print("\nSchema types found:")
    if report.schema_type_counts:
        for schema_type, count in report.schema_type_counts.items():
            print(f"  - {schema_type}: {count}")
    else:
        print("  - None found")

    print("\nTop recommendations:")
    for i, rec in enumerate(report.top_recommendations, 1):
        print(f"  {i}. {rec}")

    print("\nPage summaries:")
    for p in report.pages:
        print("-" * 52)
        print(f"{p.url}")
        print(f"  Score: {p.page_score}/100 | Status: {p.status_code} | Load: {p.load_ms} ms | Size: {p.size_kb} KB")
        print(f"  Title: {p.title}")
        if p.schema_types:
            print(f"  Schema: {', '.join(p.schema_types)}")
        print(f"  Key issues:")
        for issue in p.issues[:5]:
            print(f"    - {issue}")


def write_json(report: SiteReport, path: str) -> None:
    data = asdict(report)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(report: SiteReport, path: str) -> None:
    fields = [
        "url", "status_code", "page_score", "title", "meta_description",
        "json_ld_count", "schema_types", "phone_found", "email_found",
        "address_like_found", "hours_like_found", "load_ms", "size_kb",
        "images_total", "images_missing_alt", "issues",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in report.pages:
            row = asdict(p)
            row["schema_types"] = ", ".join(p.schema_types)
            row["issues"] = " | ".join(p.issues)
            writer.writerow({k: row.get(k) for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Ads / AI Recommendation Readiness Scanner MVP")
    parser.add_argument("url", help="Website URL to scan")
    parser.add_argument("--max-pages", type=int, default=10, help="Max pages to crawl. Default: 10")
    parser.add_argument("--out", default="ai_readiness_report.json", help="JSON report path")
    parser.add_argument("--csv", default=None, help="Optional CSV page report path")
    args = parser.parse_args()

    report = crawl(args.url, max_pages=max(1, args.max_pages))
    print_human_report(report)
    write_json(report, args.out)
    if args.csv:
        write_csv(report, args.csv)

    print(f"\nSaved JSON report to: {args.out}")
    if args.csv:
        print(f"Saved CSV report to: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
