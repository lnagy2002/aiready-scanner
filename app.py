import re
from html import escape
from urllib.parse import quote, urlparse

import pandas as pd
import streamlit as st

import ai_discovery
import ai_simulation
import datastore
from ai_readiness_scanner import comparison_summary, crawl
from report_pdf import build_pdf_report


st.set_page_config(
    page_title="LocalAgentReady",
    page_icon="✨",
    layout="wide",
)


# ---------------- Helpers ----------------

def normalize_url_for_display(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


def queue_scan_from_input() -> None:
    """Start a scan from the URL box's on_change — this is what makes pressing
    Enter in the field kick off the scan (Streamlit's Enter only commits the
    value and reruns; it doesn't click the button). Sets the same flags the
    scan button sets; the rerun that follows the callback runs the crawl."""
    url = (st.session_state.get("website_url_input") or "").strip()
    if url and not st.session_state.get("scanning"):
        st.session_state["pending_url"] = url
        st.session_state["scanning"] = True
        st.session_state["scan_error"] = ""


def pages_to_dataframe(report) -> pd.DataFrame:
    rows = []

    for p in report.pages:
        rows.append({
            "URL": p.url,
            "Status": p.status_code,
            "Score": p.page_score,
            "Title": p.title,
            "Business details": ", ".join(p.schema_types),
            "Phone": "Yes" if (p.phone_found or p.schema_has_phone) else "No",
            "Location": "Yes" if (p.address_like_found or p.schema_has_address) else "No",
            "Hours": "Yes" if (p.hours_like_found or p.schema_has_hours) else "No",
            "Top issues": " | ".join(p.issues[:4]),
        })

    return pd.DataFrame(rows)


def score_status(score: int) -> tuple[str, str]:
    if score >= 85:
        return "Strong", "strong"
    if score >= 70:
        return "Good start", "good"
    if score >= 55:
        return "Needs work", "medium"
    return "Missing key information", "low"


# A short, memorable label that sits next to the raw number. Plain "72/100"
# doesn't stick in anyone's memory or make a good screenshot; a phrase does.
# This is the same job Cruelx's "Half-Baked" label is doing for them.
def score_band(score: int) -> tuple[str, str]:
    if score >= 85:
        return "AI's Top Pick", "strong"
    if score >= 70:
        return "On AI's Radar", "good"
    if score >= 55:
        return "Easy to Miss", "medium"
    if score >= 35:
        return "Barely Visible", "low"
    return "Invisible to AI", "low"


def grade_message(score: int) -> str:
    if score >= 85:
        return "Your website gives AI tools a strong foundation to understand what you do, where you work, and why customers should trust you."
    if score >= 70:
        return "Your website has a good foundation. A few important details could be clearer so AI tools and customers understand you faster."
    if score >= 55:
        return "Your website is partly clear, but some important business details may be missing, incomplete, or hard for AI tools to read."
    if score >= 40:
        return "Your website likely needs improvement so customers and AI tools can understand your business with confidence."
    return "Your website may be hard for customers and AI tools to understand clearly. Start with the basics: services, location, trust proof, and contact details."


# Seed benchmark, used before enough real scans exist. datastore.get_typical_score
# blends this toward the true average of logged scans as volume grows (and falls
# back to it when the Google Sheet store isn't configured). The UI still labels
# the number an estimate until it's backed by meaningful volume.
TYPICAL_LOCAL_BUSINESS_SCORE = datastore.BENCHMARK_SEED

# Competitor scans use a lighter page budget than the user's own full scan —
# a comparative snapshot doesn't need every page.
COMPETITOR_MAX_PAGES = 5


def _secret(name: str, default: str = "") -> str:
    """Safe accessor for a Streamlit secret — never raises when no secrets
    file exists (e.g. a bare local run)."""
    try:
        return str(st.secrets.get(name, default)) or default
    except Exception:
        return default


@st.cache_data(ttl=300, show_spinner=False)
def current_typical_score() -> int:
    """The benchmark to display/compare against — adaptive when the scan-log
    store is configured, the seed constant otherwise. Cached briefly so it
    doesn't hit the sheet on every rerun."""
    return datastore.get_typical_score()


def is_admin() -> bool:
    """Owner-only gate for the real full-report download (#6). Reveal it by
    visiting the app with ?admin=<ADMIN_TOKEN> in the URL, matched against a
    secret. With no token configured, nobody is admin — the button stays
    hidden for everyone, which is the safe default."""
    token = _secret("ADMIN_TOKEN")
    if not token:
        return False
    try:
        supplied = st.query_params.get("admin")
    except Exception:
        supplied = None
    return supplied == token


def stripe_payment_link(report=None) -> str:
    """The $10 Stripe Payment Link, if configured. When a report is given, the
    scanned domain is passed as client_reference_id so it shows up alongside
    the payment in your Stripe dashboard — that's how you match a payment to
    the site it's for in the manual-delivery flow."""
    link = _secret("STRIPE_PAYMENT_LINK")
    if not link or report is None:
        return link
    domain = urlparse(report.normalized_url).netloc.replace("www.", "")
    if domain:
        sep = "&" if "?" in link else "?"
        link = f"{link}{sep}client_reference_id={quote(domain)}"
    return link


def homepage_of(report):
    return report.pages[0] if getattr(report, "pages", None) else None


def collect_site_gaps(report) -> list[str]:
    gaps = []
    homepage = homepage_of(report)

    if homepage:
        if not (homepage.schema_has_phone or homepage.phone_found):
            gaps.append("Your phone number may not be easy to find.")

        if not (homepage.schema_has_address or homepage.address_like_found):
            gaps.append("Your location or service area may not be clear enough.")

        if homepage.schema_has_hours or homepage.hours_like_found:
            # More nuanced than a simple yes/no. If the crawler found some hours but the site may not show a full weekly schedule,
            # we present this as partially clear instead of missing.
            gaps.append("Business hours are partly clear. Make sure the full weekly schedule is listed, including closed days.")
        else:
            gaps.append("Your business hours may be missing or hard to find.")

        if len(getattr(homepage, "service_words_found", [])) < 3:
            gaps.append("It may not be obvious which services you offer.")

        if not getattr(homepage, "has_faq_schema", False):
            gaps.append("Common customer questions may not be answered clearly.")

        if not homepage.has_local_business_schema and not homepage.has_organization_schema:
            gaps.append("Your business details may not be written in a way search engines and AI tools can easily understand.")

    if not gaps:
        gaps.append("The main business details are present. The next step is making them even clearer, more complete, and more trustworthy.")

    return gaps[:6]


# Matches the dynamic "N of M pages scanned have no structured data..."
# recommendation from build_recommendations() — its counts vary per site, so
# it can't be a plain dict lookup like the static messages below.
_SCHEMA_GAP_RE = re.compile(r"^(\d+) of (\d+) pages scanned have no structured data")


def plain_english_recommendation(rec: str) -> str:
    schema_gap = _SCHEMA_GAP_RE.match(rec)
    if schema_gap:
        missing, total = schema_gap.groups()
        return (
            f"{missing} of the {total} pages we checked are missing basic business details behind the scenes "
            "(not just the homepage) — this is likely the main reason your score is low. Make sure key pages "
            "like services, about, and contact carry the same business info as your homepage, not just the "
            "homepage itself."
        )

    replacements = {
        "Add or fix /robots.txt so crawlers can understand what can be accessed.":
            "Make sure search engines and AI tools are allowed to read your website.",

        "Add or fix /sitemap.xml and submit it in search tools.":
            "Add a simple website map so search engines and AI tools can find your important pages.",

        "Add Schema.org JSON-LD for LocalBusiness/Organization, including name, URL, logo, phone, address, service area, hours, and sameAs profiles.":
            "Add clear behind-the-scenes business details: name, services, phone, location, hours, and trusted profiles.",

        "Make the phone number visible on the homepage and include it in schema.":
            "Make your phone number easy to find on the homepage.",

        "Make the business address or service area clear on the homepage and include it in schema.":
            "Clearly show where your business is located or which areas you serve.",

        "Add clearer service and booking language: services offered, locations served, emergency/availability, quote or booking CTA.":
            "Make it clearer what you offer, where you work, and how someone can book or request a quote.",

        "Write a stronger meta description explaining who you help, where, and what you offer.":
            "Improve the short website summary that explains who you help, where you work, and what you offer.",

        "Improve image alt text so visual content is understandable to crawlers and assistive tools.":
            "Describe your important images so search engines, AI tools, and accessibility tools can understand them.",

        "Add an FAQ section with FAQPage schema for common buyer questions, pricing, timing, service area, and process.":
            "Add a helpful FAQ section that answers common customer questions.",

        "Add testimonials/reviews where appropriate; use valid review or aggregateRating schema only when it follows platform rules and reflects real reviews.":
            "Add real testimonials or review signals where appropriate to build trust.",

        "Double-check your phone number across pages — the numbers found don't clearly match, which reads as an error to AI tools and directories (not the same as legitimate multiple office lines).":
            "Double-check your phone number across pages — the numbers found don't clearly match, which can look like an error to AI tools.",

        "Make the phone number a tap-to-call (tel:) link so mobile visitors can call in one tap.":
            "Make your phone number tap-to-call on mobile, not just plain text.",

        "Link to your Google Business Profile or Google Maps listing; it's a key trust signal for local recommendations.":
            "Link to your Google Business Profile or Google Maps listing.",

        "Add a mobile viewport meta tag; most local searches and AI-assistant checks happen on phones.":
            "Make sure your website is set up to display properly on phones.",

        "Consider adding an /llms.txt file summarizing your business for AI assistants (an emerging, optional standard).":
            "Consider adding a simple AI-assistant summary file for your business (an emerging, optional standard).",
    }

    return replacements.get(rec, rec)


def get_fix_class(text: str) -> str:
    lower = text.lower()
    high_terms = ["missing", "not found", "blocked", "may not", "hard to", "not be easy", "not clear"]
    medium_terms = ["partly", "improve", "clearer", "add", "make sure", "should"]

    if any(term in lower for term in high_terms):
        return "fix-high"
    if any(term in lower for term in medium_terms):
        return "fix-medium"
    return "fix-low"


def build_fix_items(items: list[str], numbered: bool = False) -> str:
    parts = []
    for i, item in enumerate(items, start=1):
        safe_item = escape(item)
        cls = get_fix_class(item)
        prefix = f'<span class="fix-number">{i}</span>' if numbered else '<span class="fix-dot"></span>'
        parts.append(
            f'<div class="fix-item {cls}">{prefix}<div class="fix-copy">{safe_item}</div></div>'
        )
    return "".join(parts)


# Schema.org types worth showing a business owner, translated to plain
# English. Anything not in this map (BreadcrumbList, ListItem, ImageObject,
# EntryPoint, GeoCoordinates, PostalAddress, etc.) is internal wiring used to
# describe *other* schema objects — real, but meaningless to a non-technical
# reader, so it's filtered out entirely rather than shown untranslated.
SCHEMA_TYPE_LABELS = {
    "LocalBusiness": "Local business info",
    "Organization": "Business info",
    "FAQPage": "FAQ info",
    "Review": "Customer reviews",
    "AggregateRating": "Review ratings",
    "OpeningHoursSpecification": "Business hours",
    "Product": "Product info",
    "Service": "Service info",
    "PostalAddress": "Address info",
}
# Anything matching one of these substrings counts as a recognized business
# schema type even if it's a more specific subtype (e.g. "Plumber",
# "Dentist", "HVACBusiness" all count as LocalBusiness variants).
SCHEMA_TYPE_LABELS_SUBSTRING = {
    "Business": "Local business info",
    "Contractor": "Local business info",
    "Agent": "Local business info",
}


def plain_schema_label(schema_type: str) -> str | None:
    if schema_type in SCHEMA_TYPE_LABELS:
        return SCHEMA_TYPE_LABELS[schema_type]
    for key, label in SCHEMA_TYPE_LABELS_SUBSTRING.items():
        if key in schema_type:
            return label
    return None


def collect_schema_labels(report, limit: int = 6) -> list[str]:
    """Plain-English business-detail labels found across the whole site,
    shared by the on-page 'Business details found' card and the PDF."""
    labels_found: list[str] = []
    for item in report.schema_type_counts.keys():
        label = plain_schema_label(item)
        if label and label not in labels_found:
            labels_found.append(label)
    return labels_found[:limit]


def pages_detail_for_pdf(report) -> list[dict]:
    """Full per-page technical detail in the shape build_pdf_report expects
    — every issue and on-page signal, not just one issue per page like the
    free scan's page table. This is a real chunk of the paid report's
    actual value, not a reformatted copy of the free summary."""
    return [
        {
            "url": p.url,
            "score": p.page_score,
            "status": p.status_code if p.status_code is not None else "—",
            "title": p.title,
            "meta_description": p.meta_description,
            "h1": p.h1[0] if p.h1 else None,
            "schema_types": p.schema_types,
            "issues": p.issues,
            "phone_ok": bool(p.schema_has_phone or p.phone_found),
            "address_ok": bool(p.schema_has_address or p.address_like_found),
            "hours_ok": bool(p.schema_has_hours or p.hours_like_found),
            "tel_link": p.tel_link_found,
            "gbp_link": p.gbp_link_found,
            "viewport_ok": p.has_viewport_meta,
        }
        for p in report.pages
    ]


def pages_missing_schema_urls(report) -> list[str]:
    """URLs of scanned pages with zero structured data at all — the exact
    pages behind the "N of M pages..." recommendation, so the reader knows
    precisely where to start instead of guessing."""
    return [
        p.url for p in report.pages
        if p.status_code and p.status_code < 400 and p.json_ld_count == 0
    ]


def format_phone_for_snippet(raw: str | None) -> str | None:
    """Turn a found phone number (digits-only, e.g. "9252328896") into a
    presentable +1-XXX-XXX-XXXX form for the ready-to-paste schema snippet.
    Returns None if we don't have a confident 10-digit US/CA number."""
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    return f"+1-{digits[0:3]}-{digits[3:6]}-{digits[6:]}"


def mini_signal(label: str, value: str, status: str = "neutral") -> str:
    return (
        f'<div class="signal-row"><span>{escape(label)}</span>'
        f'<strong class="signal-{status}">{escape(value)}</strong></div>'
    )


# The exact same pass/fail signals comparison_summary() exposes, in display
# order — reused for both the table's rows and the "what they have that you
# don't" callout below it. Plain-English throughout on purpose: a plumber
# reading this has no reason to know what "schema" or "llms.txt" means, even
# though those are the literal field names on the SiteReport underneath.
# table_label: short, fits a narrow table column.
# callout_phrase: a full noun phrase that reads naturally after "have ___".
COMPARISON_CHECKS = [
    ("has_business_schema", "Business info AI can read", "clear behind-the-scenes business info that AI tools can read"),
    ("has_faq_schema", "Answers common questions", "a section that answers common customer questions"),
    ("has_review_schema", "Visible customer reviews", "visible customer reviews or ratings"),
    ("has_gbp_link", "Google Business Profile link", "a link to their Google Business Profile or Maps listing"),
    ("has_tel_link", "Tap-to-call phone number", "a tap-to-call phone number"),
    ("robots_found", "Lets search engines in", "a search access file so crawlers can read their site"),
    ("sitemap_found", "Website map for search engines", "a website map that helps search engines and AI tools find their pages"),
    ("llms_found", "AI-assistant summary file", "a summary file made specifically for AI assistants"),
    ("nap_consistent", "Matching phone number sitewide", "the same phone number listed on every page"),
]


def _competitor_label(url: str) -> str:
    netloc = urlparse(url).netloc.replace("www.", "")
    return netloc or url


def render_competitor_comparison(report, competitor_reports: list) -> None:
    """Backs up the homepage headline's "or your competitors" promise with
    an actual side-by-side comparison, not just a hook. Only rendered when
    the user actually supplied competitor URLs."""
    if not competitor_reports:
        return

    you = comparison_summary(report)
    competitors = [comparison_summary(r) for r in competitor_reports]
    columns = [("You", you)] + [(_competitor_label(c["url"]), c) for c in competitors]

    header_cells = "".join(f"<th>{escape(label)}</th>" for label, _ in columns)
    score_cells = "".join(f"<td><strong>{c['score']}</strong></td>" for _, c in columns)
    rows_html = ""
    for key, table_label, _callout_phrase in COMPARISON_CHECKS:
        cells = "".join(
            f'<td class="{"cmp-yes" if c[key] else "cmp-no"}">{"✓" if c[key] else "✗"}</td>'
            for _, c in columns
        )
        rows_html += f"<tr><td>{escape(table_label)}</td>{cells}</tr>"

    st.markdown('<div class="section-title">How you compare</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="signal-card">
            <table class="compare-table">
                <thead><tr><th></th>{header_cells}</tr></thead>
                <tbody>
                    <tr class="compare-score-row"><td>AgentReady Score</td>{score_cells}</tr>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gap_items = []
    for key, _table_label, callout_phrase in COMPARISON_CHECKS:
        if you[key]:
            continue
        count = sum(1 for c in competitors if c[key])
        if count:
            gap_items.append(f"{count} of {len(competitors)} competitors have {callout_phrase} — you don't.")

    if gap_items:
        st.markdown(
            f'<div class="modern-card" style="min-height: auto; margin-top: 0.9rem;">{build_fix_items(gap_items, numbered=False)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="signal-card" style="margin-top: 0.9rem;">'
            f'{mini_signal("Competitive gaps", "None — you match or beat every competitor checked", "good")}</div>',
            unsafe_allow_html=True,
        )


def render_ai_simulation(report) -> None:
    """Tests the product's actual promise directly — what a real AI
    assistant would say about this business, based only on what it can
    read on the page — instead of only inferring that from schema/meta-tag
    heuristics. Opt-in (button-triggered) since it costs a real API call
    per click; silently absent when no Anthropic API key is configured,
    rather than showing a broken or confusing feature to end users."""
    if not ai_simulation.is_available():
        return

    homepage = report.pages[0] if report.pages else None
    if not homepage or not homepage.visible_text_excerpt:
        return

    st.markdown('<div class="section-title">What would an AI assistant say about you?</div>', unsafe_allow_html=True)

    cache_key = f"ai_sim::{report.normalized_url}"

    if st.button("Simulate what AI would say about my business", key="ai_sim_button"):
        question = ai_simulation.build_customer_question(report.normalized_url)
        page_summary = ai_simulation.build_page_summary(
            title=homepage.title,
            meta_description=homepage.meta_description,
            h1=homepage.h1[0] if homepage.h1 else None,
            schema_types=homepage.schema_types,
        )
        with st.spinner("Asking an AI assistant what it would tell a customer..."):
            try:
                result = ai_simulation.simulate_ai_answer(
                    page_summary=page_summary,
                    visible_text_excerpt=homepage.visible_text_excerpt,
                    customer_question=question,
                )
                st.session_state[cache_key] = {"question": question, "result": result}
            except Exception as exc:
                st.session_state[cache_key] = None
                st.error(f"AI simulation failed: {exc}")

    cached = st.session_state.get(cache_key)
    if cached:
        question = cached["question"]
        result = cached["result"]
        confidence_class = {"High": "good", "Medium": "medium", "Low": "low"}.get(result.confidence, "neutral")

        st.markdown(
            f"""
            <div class="message-card">
                <div style="opacity: 0.65; font-size: 0.85rem; margin-bottom: 0.4rem;">Simulated customer question</div>
                <div style="margin-bottom: 1rem;">&ldquo;{escape(question)}&rdquo;</div>
                <div style="opacity: 0.65; font-size: 0.85rem; margin-bottom: 0.4rem;">What the AI assistant said</div>
                <div>{escape(result.answer)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="signal-card" style="margin-top: 0.9rem;">'
            f'{mini_signal("Confidence", result.confidence, confidence_class)}</div>',
            unsafe_allow_html=True,
        )
        if result.missing_or_unclear:
            st.markdown(
                f'<div class="modern-card" style="min-height: auto; margin-top: 0.9rem;">'
                f'{build_fix_items(result.missing_or_unclear, numbered=False)}</div>',
                unsafe_allow_html=True,
            )


def render_ai_discovery(report) -> None:
    """Runs a real AI discovery query — "who would ChatGPT recommend for a
    [type] in [location]?" — and shows whether this business appears, and
    where. Button-triggered (a live web search costs money per click) and
    silently absent when no OpenAI key is configured on the server."""
    st.markdown(
        '<div class="section-title">Does AI recommend you when customers search?</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "We ask ChatGPT — searching the live web, the way a real customer would — "
        "for the best options in your area, then check whether you show up."
    )

    if not ai_discovery.is_available():
        st.info(
            "This live check needs an OpenAI API key configured on the server. "
            "It's turned off in this environment."
        )
        return

    # Both auto-derived from the site's metadata (service type + location), so
    # there are no fields to fill in by default. The collapsed "Adjust"
    # expander is only a safety net for the rare miss (e.g. no city on the
    # page, or an unusual service name).
    default_type = ai_discovery.derive_business_type(report)
    default_loc = ai_discovery.derive_location(report)

    with st.expander("Adjust service or location (auto-detected)"):
        biz_type = st.text_input("Business type", value=default_type, key="discovery_type")
        location = st.text_input("Location (city, state, ZIP)", value=default_loc, key="discovery_location")

    if location.strip():
        st.markdown(
            f'<div class="discovery-preview">We\'ll ask ChatGPT: '
            f'<strong>&ldquo;best {escape(biz_type)} in {escape(location)}&rdquo;</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning(
            "We couldn't detect your city from the site — add it under “Adjust” above so the search is local."
        )

    cache_key = f"ai_discovery::{report.normalized_url}"

    if st.button("See if AI recommends me", key="discovery_button"):
        if not location.strip():
            st.error("Add a location (city/state) under “Adjust” so we can run a realistic local search.")
        else:
            with st.spinner("Asking ChatGPT to search the web for the best options in your area..."):
                try:
                    st.session_state[cache_key] = ai_discovery.run_discovery(report, biz_type, location)
                except Exception as exc:
                    st.session_state[cache_key] = None
                    st.error(f"AI discovery check failed: {exc}")

    result = st.session_state.get(cache_key)
    if not result:
        return

    if result.appears and result.position:
        verdict_class, verdict = "good", f"Yes — you appeared at position #{result.position}"
    elif result.appears:
        verdict_class, verdict = "good", "Yes — you were mentioned"
    else:
        verdict_class, verdict = "low", "No — you did not appear in ChatGPT's recommendations"

    st.markdown(
        f'<div class="message-card">'
        f'<div style="opacity:0.65; font-size:0.85rem; margin-bottom:0.4rem;">Simulated customer search</div>'
        f'<div style="margin-bottom:1rem;">&ldquo;{escape(result.query)}&rdquo;</div>'
        f'<div style="opacity:0.65; font-size:0.85rem; margin-bottom:0.4rem;">Result</div>'
        f'<div style="font-weight:800; font-size:1.15rem;">{escape(verdict)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if result.recommended:
        rows = ""
        for i, biz in enumerate(result.recommended, start=1):
            is_you = result.position == i
            you_tag = ' <strong class="signal-good">← this is you</strong>' if is_you else ""
            rows += f'<div class="signal-row"><span>{i}. {escape(biz.name)}{you_tag}</span></div>'
        st.markdown(
            f'<div class="section-title" style="font-size:1.05rem;">Who ChatGPT recommended</div>'
            f'<div class="signal-card">{rows}</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "A snapshot for this one query — AI answers vary by exact wording, location, and over time. "
        "Improving the fixes above is how you move from overlooked to recommended."
    )


def render_ai_discovery_teaser() -> None:
    """Locked teaser for the AI-recommendation check on the free page. The
    live check itself is a paid feature (it makes a real, billable web-search
    call), so the free view sells it rather than running it."""
    st.markdown(
        """
        <div class="section-title">Does AI recommend you when customers search?</div>
        <div class="locked-card">
            <div class="locked-head">
                <span class="locked-lock">🔒</span>
                <span class="locked-pill">Included in the full report — $10</span>
            </div>
            <p>We ask <strong>ChatGPT</strong> — searching the live web, the way a real customer would —
            <em>&ldquo;who are the best options near me?&rdquo;</em> for your service and city, then show
            whether you appear in its recommendations and at what position, alongside the competitors it names.</p>
            <p class="locked-sub">This is the clearest answer to &ldquo;is AI finding my business — or my competitors?&rdquo;</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_pdf_controls(report) -> None:
    """Owner-only: run the two live AI checks so their results get embedded in
    the downloadable PDF — WITHOUT rendering the answers on the page. The
    on-screen report stays lean; the live 'does AI recommend you?' result and
    the AI-assistant answer belong to the paid PDF. Each run is a billable API
    call, so it's button-triggered and cached per site (same session_state
    keys build_pdf_for_report reads from)."""
    st.markdown('<div class="section-title">AI checks for the PDF (owner only)</div>', unsafe_allow_html=True)
    st.caption(
        "Run these to fold the live AI-recommendation result and the AI-assistant answer into the "
        "downloaded PDF. The results appear only in the PDF, not on this page."
    )

    disc_key = f"ai_discovery::{report.normalized_url}"
    sim_key = f"ai_sim::{report.normalized_url}"
    col1, col2 = st.columns(2)

    # --- Live AI recommendation check (OpenAI web search) ---
    with col1:
        if not ai_discovery.is_available():
            st.info("Needs OPENAI_API_KEY on the server.")
        else:
            biz_type = st.text_input(
                "Business type", value=ai_discovery.derive_business_type(report), key="pdf_disc_type"
            )
            location = st.text_input(
                "Location (city, state)", value=ai_discovery.derive_location(report), key="pdf_disc_loc"
            )
            if st.button("Run AI recommendation check", key="pdf_run_discovery", use_container_width=True):
                if not location.strip():
                    st.error("Add a location so the search is local.")
                else:
                    with st.spinner("Asking ChatGPT to search the web..."):
                        try:
                            st.session_state[disc_key] = ai_discovery.run_discovery(report, biz_type, location)
                        except Exception as exc:
                            st.session_state[disc_key] = None
                            st.error(f"Failed: {exc}")
            if st.session_state.get(disc_key):
                st.success("✓ Recommendation check ready — included in the PDF.")
            else:
                st.caption("Not run yet.")

    # --- AI-assistant answer simulation (Anthropic) ---
    with col2:
        homepage = report.pages[0] if report.pages else None
        if not ai_simulation.is_available():
            st.info("Needs ANTHROPIC_API_KEY on the server.")
        elif not homepage or not homepage.visible_text_excerpt:
            st.info("No homepage text available to simulate.")
        else:
            if st.button("Run AI assistant answer", key="pdf_run_sim", use_container_width=True):
                question = ai_simulation.build_customer_question(report.normalized_url)
                page_summary = ai_simulation.build_page_summary(
                    title=homepage.title,
                    meta_description=homepage.meta_description,
                    h1=homepage.h1[0] if homepage.h1 else None,
                    schema_types=homepage.schema_types,
                )
                with st.spinner("Asking an AI assistant..."):
                    try:
                        result = ai_simulation.simulate_ai_answer(
                            page_summary=page_summary,
                            visible_text_excerpt=homepage.visible_text_excerpt,
                            customer_question=question,
                        )
                        st.session_state[sim_key] = {"question": question, "result": result}
                    except Exception as exc:
                        st.session_state[sim_key] = None
                        st.error(f"Failed: {exc}")
            if st.session_state.get(sim_key):
                st.success("✓ AI-assistant answer ready — included in the PDF.")
            else:
                st.caption("Not run yet.")


def discovery_dict_for_pdf(report) -> dict | None:
    """Shape the cached live-discovery result (if the owner ran the check) for
    the PDF's "does AI recommend you?" section. Returns None when no check has
    been run for this site, so the section is simply omitted."""
    res = st.session_state.get(f"ai_discovery::{report.normalized_url}")
    if not res:
        return None
    if res.appears and res.position:
        verdict = f"Yes — you appeared at position #{res.position} in ChatGPT's recommendations."
    elif res.appears:
        verdict = "Yes — you were mentioned in ChatGPT's answer."
    else:
        verdict = "No — you did not appear in ChatGPT's recommendations."
    recommended = [
        {"name": biz.name, "is_you": res.position == i}
        for i, biz in enumerate(res.recommended, start=1)
    ]
    # Reconcile a low on-page score with a decent live ranking: the readiness
    # score measures the site; the ranking often rides on off-site reviews.
    reconcile = ""
    if res.appears and report.site_score < 70:
        reconcile = (
            f"You appear in AI recommendations today — but largely on the strength of third-party review "
            f"sites, not your own website (your AI-readiness score is {report.site_score}/100). Strengthening "
            f"the fixes in this report is how you climb the ranking and stop depending on aggregators."
        )

    return {
        "query": res.query,
        "appears": res.appears,
        "position": res.position,
        "verdict": verdict,
        "recommended": recommended,
        "answer_text": res.answer_text,
        "reconcile": reconcile,
    }


def ai_answer_dict_for_pdf(report) -> dict | None:
    """Shape the cached AI-assistant simulation (if run) for the PDF."""
    cached = st.session_state.get(f"ai_sim::{report.normalized_url}")
    if not cached:
        return None
    result = cached["result"]
    return {
        "question": cached["question"],
        "answer": result.answer,
        "confidence": result.confidence,
        "missing": result.missing_or_unclear,
    }


def build_pdf_for_report(report) -> bytes:
    """Assemble the real full-report PDF from a just-completed scan."""
    status_label, _ = score_status(report.site_score)
    band_label, _ = score_band(report.site_score)

    typical = current_typical_score()
    delta = report.site_score - typical
    benchmark_text = (
        f"{'+' if delta >= 0 else ''}{delta} vs. the ~{typical}/100 "
        "typical local-business site we see (early estimate, not a large-scale average yet)."
    )

    recs = [plain_english_recommendation(r) for r in report.top_recommendations]
    # Covers both cases the scanner can report: no business schema anywhere,
    # or schema present on only some pages (the far more common real-world
    # pattern — see the "N of M pages..." recommendation) — either way, the
    # reader needs the ready-to-paste snippet to close the gap.
    include_snippet = any(
        "Schema.org JSON-LD" in r or "no structured data (JSON-LD) at all" in r
        for r in report.top_recommendations
    )
    phone = format_phone_for_snippet(report.phone_numbers_found[0] if report.phone_numbers_found else None)

    return build_pdf_report(
        site_label=report.normalized_url,
        scanned_at=report.scanned_at,
        score=report.site_score,
        grade=report.grade,
        band_label=band_label,
        message=grade_message(report.site_score),
        benchmark_text=benchmark_text,
        recommendations=recs,
        gaps=collect_site_gaps(report),
        robots_found=bool(report.robots_status and report.robots_status < 400),
        sitemap_found=bool(report.sitemap_status and report.sitemap_status < 400),
        llms_found=bool(report.llms_txt_status and report.llms_txt_status < 400),
        schema_labels=collect_schema_labels(report),
        pages_rows=pages_detail_for_pdf(report),
        missing_schema_urls=pages_missing_schema_urls(report),
        category_scores=compute_category_scores(report),
        snippet_url=report.normalized_url,
        snippet_phone=phone,
        include_schema_snippet=include_snippet,
        discovery=discovery_dict_for_pdf(report),
        ai_answer=ai_answer_dict_for_pdf(report),
    )


def render_report_purchase(report) -> None:
    """Visitor-facing purchase step: collect buyer details, log the lead,
    and send them to Stripe checkout when a payment link is configured."""
    purchase_intro = (
        '<div class="purchase-panel">'
        '<div class="purchase-panel-head">'
        '<div>'
        '<div class="purchase-eyebrow">Personalized PDF report</div>'
        '<h2>Get your full report — $10</h2>'
        '<p>Enter your details below. After checkout, we’ll email the personalized PDF report to the address you provide.</p>'
        '</div>'
        '<div class="purchase-price-badge">$10</div>'
        '</div>'
        '<div class="purchase-trust-row">'
        '<span>Secure checkout</span>'
        '<span>Manual PDF delivery</span>'
        '<span>Uses this scan’s results</span>'
        '</div>'
        '</div>'
    )
    st.markdown(purchase_intro, unsafe_allow_html=True)

    # No st.form here on purpose: a form would need a submit click (server
    # round-trip) to save the lead, and THEN a second click to reach Stripe.
    # Plain inputs rerun as each field commits (blur/Enter), so we can save the
    # lead as it's filled and make the checkout a single real link click.
    with st.container(key="report_purchase_form"):
        pc1, pc2 = st.columns(2, gap="medium")
        with pc1:
            st.markdown('<div class="field-label">Company name</div>', unsafe_allow_html=True)
            company = st.text_input("Company name", label_visibility="collapsed", key="purchase_company")
            st.markdown('<div class="field-label">Phone number</div>', unsafe_allow_html=True)
            phone = st.text_input("Phone number", label_visibility="collapsed", key="purchase_phone")
        with pc2:
            st.markdown('<div class="field-label">Contact name</div>', unsafe_allow_html=True)
            contact_name = st.text_input("Contact name", label_visibility="collapsed", key="purchase_contact_name")
            st.markdown('<div class="field-label">Email address</div>', unsafe_allow_html=True)
            email = st.text_input("Email address", label_visibility="collapsed", key="purchase_email")

        values = {"company": company, "contact_name": contact_name, "phone": phone, "email": email}
        all_filled = all(v.strip() for v in values.values())
        link = stripe_payment_link(report)

        if not all_filled:
            # Placeholder until every field is filled — so the one real click on
            # the live button below always has the lead already captured.
            st.markdown(
                '<div class="checkout-button checkout-button-disabled">'
                'Fill in your details to continue — $10</div>',
                unsafe_allow_html=True,
            )
            return

        # All present: save the lead now (before the click), de-duped by content
        # so unrelated reruns don't append duplicate rows. Sheet-only — the leads
        # worksheet is the source of truth (see datastore.log_lead).
        sig = (company, contact_name, phone, email, report.normalized_url)
        if st.session_state.get("purchase_logged_sig") != sig:
            st.session_state["purchase_logged_ok"] = datastore.log_lead({
                **values,
                "website": report.normalized_url,
                "score": report.site_score,
                "grade": report.grade,
            })
            st.session_state["purchase_logged_sig"] = sig

        if link:
            st.markdown(
                f'<a class="checkout-button" href="{escape(link)}" target="_blank" rel="noopener">'
                'Continue to secure checkout — $10 →</a>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="admin-note">We’ll email your report to {escape(email)} after checkout.</div>',
                unsafe_allow_html=True,
            )
        else:
            # No payment link configured yet — still captured the request.
            st.markdown(
                '<div class="form-message form-message-success">'
                f'<strong>Thanks, {escape(contact_name)}.</strong> We received your request and will email your report to <strong>{escape(email)}</strong> shortly.'
                '</div>',
                unsafe_allow_html=True,
            )

        if is_admin():
            st.markdown(
                '<div class="admin-note">Owner view — lead logged to sheet: '
                f'{escape(str(st.session_state.get("purchase_logged_ok")))}</div>',
                unsafe_allow_html=True,
            )


def render_full_report_preview(report) -> None:
    """Right-hand CTA in the full-report section. Two very different things
    depending on who's looking:
      - Owner (admin, via ?admin=<token>): the real $10 PDF download, built
        from this scan — the only place it's ever exposed (#6).
      - Everyone else: a "Get the full report — $10" button to the Stripe
        payment link, since delivery is manual (you email the PDF after
        payment). Falls back to pointing at the request form if no link is set."""
    if is_admin():
        pdf_bytes = build_pdf_for_report(report)
        st.download_button(
            "Download full report (PDF) ↓",
            data=pdf_bytes,
            file_name="localagentready_full_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="full_report_pdf_link",
        )
        st.caption("Owner view — this download is hidden from visitors.")
        return

    link = stripe_payment_link(report)
    if link:
        st.markdown(
            f'<a href="{escape(link)}" target="_blank" rel="noopener" '
            'style="display:flex;align-items:center;justify-content:center;height:58px;'
            'background:var(--accent);color:#ffffff;font-weight:820;font-size:1.03rem;'
            'border-radius:0.5rem;text-decoration:none;">Get the full report — $10 ↗</a>',
            unsafe_allow_html=True,
        )
        st.caption("Secure checkout via Stripe. We email your personalized PDF after payment.")
    else:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:center;height:58px;'
            'background:var(--soft);border:1.5px solid var(--line);color:var(--ink);'
            'font-weight:820;border-radius:0.5rem;">Request the full report below ↓</div>',
            unsafe_allow_html=True,
        )


# Fabricated end-to-end example so a first-time visitor can see the shape of
# a full report without it being tied to any real scanned site. Every number
# and issue below is made up for illustration — it must never be presented as
# real aggregate data (see TYPICAL_LOCAL_BUSINESS_SCORE for the same rule).
FAKE_REPORT_SCORE = 61
FAKE_REPORT_GRADE = "C"
FAKE_REPORT_SITE_LABEL = "https://example-hvac-plumbing.com"
FAKE_REPORT_SCANNED_AT = "Example only — not a real scan"
FAKE_REPORT_PHONE_RAW = "9255550142"  # digits-only, same shape a real scan finds
FAKE_REPORT_RECOMMENDATIONS = [
    "4 of 5 pages scanned have no structured data (JSON-LD) at all — only the homepage carries your business "
    "schema. This is usually the single biggest thing holding a site's score down. Add at least basic "
    "Organization/LocalBusiness JSON-LD to your other key pages too.",
    "Add a helpful FAQ section that answers common customer questions.",
    "Make your phone number tap-to-call on mobile, not just plain text.",
    "Double-check your phone number across pages — the numbers found don't clearly match, which can look like an error to AI tools.",
    "Link to your Google Business Profile or Google Maps listing.",
]
FAKE_REPORT_GAPS = [
    "Business hours are partly clear. Make sure the full weekly schedule is listed, including closed days.",
    "It may not be obvious which services you offer.",
    "Common customer questions may not be answered clearly.",
]
FAKE_REPORT_SCHEMA_LABELS = ["Local business info", "Business hours"]
FAKE_REPORT_CATEGORY_SCORES = [
    {"title": "AI Crawler Access", "score": 85, "status_label": "Good", "description": "robots.txt and a sitemap are present; adding an llms.txt file would complete the setup."},
    {"title": "Structured Data", "score": 55, "status_label": "Needs work", "description": "Only the homepage carries business schema — the other pages read as blank to AI tools. No FAQ or review schema."},
    {"title": "Content Structure", "score": 60, "status_label": "Needs work", "description": "Thin homepage content and a missing heading on one page make it harder for AI to extract expertise."},
    {"title": "Meta & SEO Tags", "score": 70, "status_label": "Good", "description": "Title and social tags are present, but some pages are missing a meta description."},
    {"title": "Technical Performance", "score": 65, "status_label": "Needs work", "description": "HTTPS is in place; no PageSpeed/Core Web Vitals data available for a deeper check."},
    {"title": "Authority & Trust", "score": 45, "status_label": "Needs work", "description": "Few verifiable trust signals — no review schema or Google Business Profile link found."},
]
FAKE_REPORT_MISSING_SCHEMA_URLS = [
    "https://example-hvac-plumbing.com/services",
    "https://example-hvac-plumbing.com/contact",
    "https://example-hvac-plumbing.com/about",
    "https://example-hvac-plumbing.com/blog/furnace-tips",
]
FAKE_REPORT_PAGES = [
    {
        "url": "https://example-hvac-plumbing.com/", "score": 74, "status": 200,
        "title": "Example HVAC & Plumbing | 24/7 Service", "meta_description": "Fast, licensed HVAC and plumbing repair in your area.",
        "h1": "24/7 HVAC & Plumbing Repair", "schema_types": ["LocalBusiness", "PostalAddress"],
        "issues": ["No FAQPage schema found.", "No review or rating schema found."],
        "phone_ok": True, "address_ok": True, "hours_ok": True, "tel_link": True, "gbp_link": False, "viewport_ok": True,
    },
    {
        "url": "https://example-hvac-plumbing.com/services", "score": 68, "status": 200,
        "title": "Our Services", "meta_description": None, "h1": "Our Services", "schema_types": [],
        "issues": ["Meta description is missing, too short, or too long.", "No JSON-LD structured data found."],
        "phone_ok": False, "address_ok": False, "hours_ok": False, "tel_link": False, "gbp_link": False, "viewport_ok": True,
    },
    {
        "url": "https://example-hvac-plumbing.com/contact", "score": 52, "status": 200,
        "title": "Contact Us", "meta_description": "Get in touch with our team.", "h1": "Contact Us", "schema_types": [],
        "issues": [
            "No address signal found.", "No JSON-LD structured data found.",
            "No click-to-call (tel:) link found; mobile visitors have to copy the number manually.",
        ],
        "phone_ok": False, "address_ok": False, "hours_ok": False, "tel_link": False, "gbp_link": False, "viewport_ok": True,
    },
    {
        "url": "https://example-hvac-plumbing.com/about", "score": 61, "status": 200,
        "title": "About Example HVAC", "meta_description": "Learn about our family-owned business.", "h1": "About Us",
        "schema_types": [], "issues": ["No review or rating schema found.", "No JSON-LD structured data found."],
        "phone_ok": False, "address_ok": False, "hours_ok": False, "tel_link": False, "gbp_link": False, "viewport_ok": True,
    },
    {
        "url": "https://example-hvac-plumbing.com/blog/furnace-tips", "score": 49, "status": 200,
        "title": "furnace tips", "meta_description": None, "h1": None, "schema_types": [],
        "issues": ["Title tag is missing, too short, or too long.", "No H1 found.", "No JSON-LD structured data found."],
        "phone_ok": False, "address_ok": False, "hours_ok": False, "tel_link": False, "gbp_link": False, "viewport_ok": True,
    },
]


@st.cache_data(show_spinner=False)
def build_fake_example_pdf() -> bytes:
    """The fabricated report behind the "EXAMPLE" link. Cached since the
    fake data never changes — no reason to re-render it on every rerun."""
    band_label, _ = score_band(FAKE_REPORT_SCORE)
    return build_pdf_report(
        site_label=FAKE_REPORT_SITE_LABEL,
        scanned_at=FAKE_REPORT_SCANNED_AT,
        score=FAKE_REPORT_SCORE,
        grade=FAKE_REPORT_GRADE,
        band_label=band_label,
        message=grade_message(FAKE_REPORT_SCORE),
        benchmark_text=(
            f"+{FAKE_REPORT_SCORE - TYPICAL_LOCAL_BUSINESS_SCORE} vs. the "
            f"~{TYPICAL_LOCAL_BUSINESS_SCORE}/100 typical local-business site we see."
        ),
        recommendations=FAKE_REPORT_RECOMMENDATIONS,
        gaps=FAKE_REPORT_GAPS,
        robots_found=True,
        sitemap_found=True,
        llms_found=False,
        schema_labels=FAKE_REPORT_SCHEMA_LABELS,
        pages_rows=FAKE_REPORT_PAGES,
        missing_schema_urls=FAKE_REPORT_MISSING_SCHEMA_URLS,
        category_scores=FAKE_REPORT_CATEGORY_SCORES,
        snippet_url=FAKE_REPORT_SITE_LABEL,
        snippet_phone=format_phone_for_snippet(FAKE_REPORT_PHONE_RAW),
        include_schema_snippet=True,
        watermark="Example — fabricated sample data, not a real scan",
    )


# ---------------- Category sub-scores ----------------

# Six categories derived from data the scanner already collects, so the
# results page can present a structured breakdown (like the reference
# layout) instead of one flat number plus a long list. Rule-based and
# deterministic — no API cost — and honest about what we can't measure
# (e.g. we have no PageSpeed/Core Web Vitals data, so Technical
# Performance says so rather than inventing a number).

def _cat_status(score: int) -> tuple[str, str]:
    if score >= 70:
        return "Good", "good"
    if score >= 45:
        return "Needs work", "medium"
    return "Poor", "low"


def _approx_word_count(report) -> int:
    hp = homepage_of(report)
    if not hp or not hp.visible_text_excerpt:
        return 0
    return len(hp.visible_text_excerpt.split())


def compute_category_scores(report) -> list[dict]:
    hp = homepage_of(report)
    https = report.normalized_url.lower().startswith("https")
    robots = bool(report.robots_status and report.robots_status < 400)
    sitemap = bool(report.sitemap_status and report.sitemap_status < 400)
    llms = bool(report.llms_txt_status and report.llms_txt_status < 400)

    any_json_ld = any(p.json_ld_count for p in report.pages)
    has_business = any(p.has_local_business_schema or p.has_organization_schema for p in report.pages)
    has_faq = any(p.has_faq_schema for p in report.pages)
    has_review = any(p.has_review_or_rating_schema for p in report.pages)
    has_gbp = any(p.gbp_link_found for p in report.pages)
    word_count = _approx_word_count(report)

    def clamp(n: int) -> int:
        return max(0, min(100, n))

    cats: list[dict] = []

    # --- AI Crawler Access ---
    score = (40 if robots else 0) + (45 if sitemap else 0) + (15 if llms else 0)
    have = [n for n, ok in (("robots.txt", robots), ("a sitemap", sitemap), ("an llms.txt", llms)) if ok]
    missing = [n for n, ok in (("robots.txt", robots), ("a sitemap", sitemap), ("an llms.txt file", llms)) if not ok]
    desc = f"Your site has {', '.join(have)}." if have else "Crawlers have little to work with here."
    if missing:
        desc += f" Still missing: {', '.join(missing)}. These help AI tools and search engines find and read every page."
    else:
        desc += " This is an ideal setup for AI tools to discover and read your pages."
    cats.append({"key": "crawler", "title": "AI Crawler Access", "icon": "🔍", "score": clamp(score), "description": desc})

    # --- Structured Data ---
    # Coverage matters as much as presence: schema on only the homepage
    # while every other page has none is a major gap (and the biggest driver
    # of a low overall score), so weight site-wide coverage heavily here —
    # otherwise this category reads "green" while the headline number is low.
    valid_pages = [p for p in report.pages if p.status_code and p.status_code < 400]
    pages_with_schema = [p for p in valid_pages if p.json_ld_count > 0]
    coverage = len(pages_with_schema) / max(len(valid_pages), 1)
    score = round(
        (20 if any_json_ld else 0)
        + 30 * coverage
        + (20 if has_business else 0)
        + (15 if has_faq else 0)
        + (15 if has_review else 0)
    )
    found = collect_schema_labels(report)
    if found:
        desc = f"Structured business data found: {', '.join(found)}."
    else:
        desc = "No structured business data (JSON-LD) was found — AI tools have to guess what your business is."
    if any_json_ld and len(pages_with_schema) < len(valid_pages):
        desc += (
            f" But only {len(pages_with_schema)} of {len(valid_pages)} pages carry it — the rest read as "
            "blank to AI tools, which may land on any page."
        )
    missing_schema = []
    if not has_faq:
        missing_schema.append("FAQ")
    if not has_review:
        missing_schema.append("reviews/ratings")
    if missing_schema:
        desc += f" Adding {', '.join(missing_schema)} schema would help AI cite you for specific questions and trust signals."
    cats.append({"key": "schema", "title": "Structured Data", "icon": "🧩", "score": clamp(score), "description": desc})

    # --- Content Structure ---
    h1_count = len(hp.h1) if hp else 0
    score = 0
    score += 25 if h1_count == 1 else (10 if h1_count >= 1 else 0)
    score += 30 if word_count >= 800 else (18 if word_count >= 400 else (5 if word_count else 0))
    score += 20 if has_faq else 0
    score += 25 if (hp and len(hp.service_words_found) >= 3) else (10 if (hp and hp.service_words_found) else 0)
    desc_parts = []
    if h1_count > 1:
        desc_parts.append(f"the homepage has {h1_count} H1 headings (one clear H1 reads better to AI)")
    elif h1_count == 0:
        desc_parts.append("no clear main heading (H1) was found")
    if word_count:
        desc_parts.append(f"roughly {word_count} words of homepage content")
    desc = "Content depth: " + ("; ".join(desc_parts) + "." if desc_parts else "limited signals found.")
    if word_count and word_count < 400:
        desc += " Thin content makes it hard for AI to extract real expertise about your services."
    cats.append({"key": "content", "title": "Content Structure", "icon": "📝", "score": clamp(score), "description": desc})

    # --- Meta & SEO Tags ---
    score = 0
    score += 30 if (hp and 20 <= hp.title_len <= 70) else 0
    score += 30 if (hp and 50 <= hp.meta_description_len <= 170) else 0
    score += 20 if (hp and hp.og_tags >= 3) else (10 if (hp and hp.og_tags) else 0)
    score += 20 if (hp and hp.has_viewport_meta) else 0
    meta_missing = []
    if not (hp and 20 <= hp.title_len <= 70):
        meta_missing.append("a well-sized page title")
    if not (hp and 50 <= hp.meta_description_len <= 170):
        meta_missing.append("a strong meta description")
    if not (hp and hp.og_tags >= 3):
        meta_missing.append("social preview tags")
    if meta_missing:
        desc = "Your page's summary tags need work: missing " + ", ".join(meta_missing) + "."
    else:
        desc = "Title, description, and social preview tags are all in good shape for search and sharing."
    cats.append({"key": "meta", "title": "Meta & SEO Tags", "icon": "🏷️", "score": clamp(score), "description": desc})

    # --- Technical Performance ---
    load = hp.load_ms if hp else None
    size = hp.size_kb if hp else None
    score = 40 if https else 0
    score += 35 if (load is not None and load <= 2500) else (15 if load is not None else 0)
    score += 25 if (size is not None and size <= 1500) else (10 if size is not None else 0)
    desc = ("Your site uses HTTPS" if https else "Your site is not using HTTPS, which hurts trust and indexing")
    if load is not None:
        desc += f" and the homepage responded in about {load} ms in our simple test"
    desc += ". We don't have full PageSpeed/Core Web Vitals data — a deeper speed audit is worth running separately."
    cats.append({"key": "technical", "title": "Technical Performance", "icon": "⚡", "score": clamp(score), "description": desc})

    # --- Authority & Trust ---
    # Uses the scanner's authority_score directly, so this category equals the
    # trust component that now weighs on the headline score (a site with no
    # trust signals can't near-100). On-page proxies only — real off-page
    # reputation (review volume, Google Business Profile prominence, directory
    # presence) needs external data sources not measured here.
    any_social = bool(hp and (hp.schema_has_same_as or hp.social_links))
    trust = []
    if has_gbp:
        trust.append("a Google Business Profile / Maps link")
    if has_review:
        trust.append("review or rating schema")
    if any_social:
        trust.append("linked social/profile pages")
    desc = ("Trust signals found: " + ", ".join(trust) + "." if trust else "Almost no trust signals were found on the site.")
    if not (has_review and has_gbp):
        desc += (
            " AI assistants recommend businesses they can verify — add review/rating schema and link your "
            "Google Business Profile. (This measures on-page trust signals; your real review volume and "
            "directory presence off-site matter too.)"
        )
    cats.append({"key": "authority", "title": "Authority & Trust", "icon": "⭐",
                 "score": clamp(report.authority_score), "description": desc})

    for c in cats:
        label, cls = _cat_status(c["score"])
        c["status_label"] = label
        c["status_class"] = cls
    return cats


def overall_summary(report) -> list[str]:
    """Plain-English summary for the business owner — no jargon, no
    category names. Leads with where they stand, names the single most
    important fix in everyday language, and states the stakes."""
    bullets = [grade_message(report.site_score)]
    if report.top_recommendations:
        bullets.append(
            "The most important fix right now: "
            + plain_english_recommendation(report.top_recommendations[0])
        )
    # Stakes line, readiness-vs-visibility aware. If a live discovery check has
    # run and the business already appears, the "AI overlooks you" line would be
    # provably false — so reframe: they're being carried by off-site reviews, and
    # a weak site is what lets competitors overtake them.
    disc = st.session_state.get(f"ai_discovery::{report.normalized_url}")
    appeared = bool(disc and getattr(disc, "appears", False))
    if appeared and report.site_score < 70:
        bullets.append(
            "You already appear in AI recommendations — but largely on the strength of third-party review "
            "sites, not your own website. Strengthening these fixes is how you climb the ranking and stop "
            "depending on aggregators that can change at any time."
        )
    elif report.site_score < 70:
        bullets.append(
            "Until this is addressed, AI tools may overlook your business and recommend a competitor instead."
        )
    else:
        bullets.append(
            "You're in good shape — a few refinements will make you even easier for AI tools to recommend with confidence."
        )
    return bullets


def score_gauge_svg(score: int, tier_class: str) -> str:
    color = {
        "strong": "var(--accent)",
        "good": "var(--accent)",
        "medium": "var(--tier-medium-text)",
        "low": "var(--tier-low-text)",
    }.get(tier_class, "var(--accent)")
    radius = 54
    circumference = 2 * 3.1415926535 * radius
    offset = circumference * (1 - max(0, min(100, score)) / 100)
    return f"""
    <svg class="gauge" viewBox="0 0 130 130" width="150" height="150">
        <circle cx="65" cy="65" r="{radius}" fill="none" stroke="var(--line)" stroke-width="11"/>
        <circle cx="65" cy="65" r="{radius}" fill="none" stroke="{color}" stroke-width="11"
                stroke-linecap="round" stroke-dasharray="{circumference:.1f}"
                stroke-dashoffset="{offset:.1f}" transform="rotate(-90 65 65)"/>
        <text x="65" y="60" text-anchor="middle" class="gauge-num">{score}</text>
        <text x="65" y="80" text-anchor="middle" class="gauge-den">/ 100</text>
    </svg>
    """


def render_category_grid(cats: list[dict]) -> None:
    # NB: keep this HTML free of leading indentation — Streamlit's markdown
    # renderer treats 4-space-indented lines as code blocks, which would
    # mangle every card after the first. Build it as one flat string.
    cards = ""
    for c in cats:
        cls = c["status_class"]
        cards += (
            '<div class="category-card">'
            '<div class="cat-head">'
            f'<div class="cat-icon">{c["icon"]}</div>'
            '<div class="cat-titles">'
            f'<div class="cat-title">{escape(c["title"])}</div>'
            f'<div class="cat-status cat-status-{cls}">{escape(c["status_label"])}</div>'
            '</div>'
            f'<div class="cat-score cat-status-{cls}">{c["score"]}</div>'
            '</div>'
            f'<div class="cat-desc">{escape(c["description"])}</div>'
            f'<div class="cat-bar"><div class="cat-bar-fill cat-fill-{cls}" style="width: {c["score"]}%;"></div></div>'
            '</div>'
        )
    st.markdown(f'<div class="category-grid">{cards}</div>', unsafe_allow_html=True)


def render_results_page(report, competitor_reports: list) -> None:
    """Dedicated results view: score gauge, polished overall summary,
    prioritized fixes, competitor comparison, and full-report CTA."""

    if st.button("← Check another site", key="back_button"):
        st.session_state["view"] = "home"
        st.rerun()

    band_label, band_class = score_band(report.site_score)
    site_label = report.normalized_url.replace("https://", "").replace("http://", "").rstrip("/")

    st.markdown(f'<div class="results-site">{escape(site_label)}</div>', unsafe_allow_html=True)

    axis_panels_html = (
        '<div class="axis-panel-wrap">'
        '<div class="axis-panel-header">Two things decide whether AI sends you customers</div>'
        '<div class="axis-panel-grid">'

        '<div class="axis-panel axis-panel-free">'
        '<div class="axis-panel-top">'
        '<div class="axis-panel-kicker-wrap">'
        '<span class="axis-panel-number axis-panel-number-free">1</span>'
        '<span class="axis-panel-kicker">Included free</span>'
        '</div>'
        f'<span class="axis-panel-status cat-status-{band_class}">{escape(band_label)}</span>'
        '</div>'
        '<div class="axis-panel-main">'
        '<div>'
        '<h3>Website AI-Readiness</h3>'
        '<p>How clearly AI tools can read and understand your website.</p>'
        '</div>'
        '</div>'
        '<div class="axis-score-row">'
        f'<div class="axis-score"><span class="axis-score-num cat-status-{band_class}">{report.site_score}</span><span class="axis-score-den">/100</span></div>'
        '<div class="axis-score-divider"></div>'
        f'<div class="axis-score-copy">{escape(grade_message(report.site_score))}</div>'
        '</div>'
        '</div>'

        '<div class="axis-panel axis-panel-paid">'
        '<div class="axis-panel-top">'
        '<div class="axis-panel-kicker-wrap">'
        '<span class="axis-panel-number axis-panel-number-paid">2</span>'
        '<span class="axis-panel-kicker">Paid add-on</span>'
        '</div>'
        '<span class="axis-panel-lock">Full report</span>'
        '</div>'
        '<div class="axis-panel-main">'
        '<div>'
        '<h3>AI Visibility</h3>'
        '<p>We check whether AI tools actually recommend your business when customers search.</p>'
        '</div>'
        '</div>'
        '<ul class="axis-panel-list">'
        '<li>See if AI recommends you or a competitor</li>'
        '<li>Get the prompt and result snapshot</li>'
        '</ul>'
        '<div class="axis-panel-action">'
        '<a class="axis-panel-button" href="#full-report">Get full report →</a>'
        '<div class="axis-panel-note">🔒 Includes full findings, competitor insights, and step-by-step fixes.</div>'
        '</div>'
        '</div>'

        '</div>'
        '</div>'
    )
    st.markdown(axis_panels_html, unsafe_allow_html=True)

    # Benchmark context for the score (adaptive — see current_typical_score).
    typical = current_typical_score()
    delta = report.site_score - typical
    delta_class = "benchmark-ahead" if delta >= 0 else "benchmark-behind"
    delta_text = f"+{delta} above typical" if delta >= 0 else f"{delta} below typical"

    # ---- Overall summary ----
    # Build as one clean, unindented HTML string. Leading indentation in a
    # triple-quoted block can make Streamlit render the HTML as a code block.
    summary = overall_summary(report)
    where_you_stand = summary[0] if len(summary) > 0 else grade_message(report.site_score)
    main_fix = (
        summary[1].replace("The most important fix right now: ", "")
        if len(summary) > 1
        else "Start with the most important missing business details."
    )
    why_it_matters = (
        summary[2]
        if len(summary) > 2
        else "Clearer business information helps AI tools understand and recommend your business with more confidence."
    )

    # IMPORTANT:
    # Streamlit Markdown can accidentally render indented multi-line HTML as a
    # code block. Keep this HTML as one continuous string with no blank/indented
    # HTML lines.
    summary_html = (
        '<div class="summary-panel">'
        '<div class="summary-top">'
        '<div>'
        '<div class="summary-eyebrow">Your AI-readiness snapshot</div>'
        '<h2>Overall summary</h2>'
        '<p class="summary-subtitle">What your score means, what is holding the site back, and why it matters for AI recommendations.</p>'
        '</div>'
        f'<div class="summary-score-pill cat-status-{band_class}">{escape(band_label)} · {report.site_score}/100</div>'
        '</div>'
        '<div class="summary-insight-grid">'
        '<div class="summary-insight-card">'
        '<div class="summary-insight-kicker">01 · Where you stand</div>'
        f'<p>{escape(where_you_stand)}</p>'
        '</div>'
        '<div class="summary-insight-card">'
        '<div class="summary-insight-kicker">02 · Most important fix</div>'
        f'<p>{escape(main_fix)}</p>'
        '</div>'
        '<div class="summary-insight-card">'
        '<div class="summary-insight-kicker">03 · Why it matters</div>'
        f'<p>{escape(why_it_matters)}</p>'
        '</div>'
        '</div>'
        '<div class="summary-benchmark">'
        '<div>'
        '<strong>How you stack up</strong>'
        f'<span>Most local business sites we check score around {typical}/100. Early estimate — not a large-scale average yet.</span>'
        '</div>'
        f'<div class="benchmark-delta {delta_class}">{escape(delta_text)}</div>'
        '</div>'
        '</div>'
    )
    st.markdown(summary_html, unsafe_allow_html=True)

    # ---- Free scan results ----
    left, right = st.columns(2, gap="medium")
    with left:
        recs = [plain_english_recommendation(rec) for rec in report.top_recommendations[:5]]
        st.markdown(
            f'<div class="result-card" style="min-height: auto;">'
            f'<div class="card-title">Top things to improve</div>'
            f'<div class="card-underline"></div>'
            f'{build_fix_items(recs, numbered=True)}</div>',
            unsafe_allow_html=True,
        )
    with right:
        gaps = collect_site_gaps(report)
        st.markdown(
            f'<div class="result-card" style="min-height: auto;">'
            f'<div class="card-title">What may not be clear</div>'
            f'<div class="card-underline"></div>'
            f'{build_fix_items(gaps, numbered=False)}</div>',
            unsafe_allow_html=True,
        )

    # ---- Mid-page CTA: jump to the full-report section ----
    # Anchors to #full-report (target rendered just above that section below).
    st.markdown(
        '<div class="fix-cta">'
        '<div class="fix-cta-icon">📄</div>'
        '<div class="fix-cta-copy">'
        '<strong>Want the exact fixes?</strong>'
        '<span>Get the full PDF: your AI visibility (does AI recommend you?), page-by-page details, and ready-to-paste fixes.</span>'
        '</div>'
        '<a class="fix-cta-button" href="#full-report">Get the full fix plan — $10 →</a>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Featured competitor section ----
    competitor_html = """
<div class="competitor-feature">
  <div class="competitor-copy-block">
    <div class="competitor-eyebrow">See what they have that you don't</div>
    <h2>Compare with competitors</h2>
    <p>Add up to 3 competitor websites. We'll compare their AI-readiness signals against yours and show where they may be easier for AI tools to understand, trust, or recommend.</p>
  </div>

  <div class="competitor-micro-grid">
    <div class="competitor-micro">
      <div class="competitor-micro-icon">01</div>
      <strong>Services & locations</strong>
      <span>See who explains their offerings and service areas best.</span>
    </div>
    <div class="competitor-micro">
      <div class="competitor-micro-icon">02</div>
      <strong>Trust signals</strong>
      <span>Compare reviews, testimonials, profile links, and credibility builders.</span>
    </div>
    <div class="competitor-micro">
      <div class="competitor-micro-icon">03</div>
      <strong>Customer answers</strong>
      <span>See who answers common questions most clearly.</span>
    </div>
  </div>
</div>
"""
    st.markdown(competitor_html, unsafe_allow_html=True)

    st.markdown(
        '<div class="competitor-form-intro">Enter competitor websites below, then run the comparison.</div>',
        unsafe_allow_html=True,
    )
    cc1, cc2, cc3 = st.columns(3, gap="small")
    with cc1:
        cu1 = st.text_input("Competitor 1", placeholder="https://competitor1.com", key="competitor_url_1")
    with cc2:
        cu2 = st.text_input("Competitor 2", placeholder="https://competitor2.com", key="competitor_url_2")
    with cc3:
        cu3 = st.text_input("Competitor 3", placeholder="https://competitor3.com", key="competitor_url_3")

    if st.button("Compare", key="run_compare"):
        urls = [u.strip() for u in (cu1, cu2, cu3) if u.strip()]
        reports = []
        if urls:
            with st.spinner(f"Checking {len(urls)} competitor site(s)..."):
                for u in urls:
                    try:
                        reports.append(crawl(normalize_url_for_display(u), max_pages=COMPETITOR_MAX_PAGES))
                    except Exception:
                        continue
        st.session_state["competitor_reports"] = reports
        st.rerun()

    render_competitor_comparison(report, competitor_reports)

    # ---- Does AI recommend you? ----
    # Hidden from visitors. The owner (admin) gets the controls to run the live
    # AI checks — their results are folded into the downloadable PDF (#2), not
    # shown on this page.
    if is_admin():
        render_ai_pdf_controls(report)

    # ---- Full report & download ----
    full_report_html = (
        '<div class="full-picture-section">'
        '<div class="full-picture-copy">'
        '<div class="full-picture-eyebrow">Full report</div>'
        '<h2>Want the full picture?</h2>'
        '<p class="full-picture-lede">Your score above measures your website\'s <strong>AI-readiness</strong>. The full report goes further — it checks your real <strong>AI visibility</strong> (does ChatGPT actually recommend you?), plus a prioritized fix plan, ready-to-paste schema, and page-by-page detail as a PDF.</p>'
        '<div class="full-feature-grid">'
        '<div class="full-feature"><span>01</span><strong>Live AI recommendation check</strong><p>We ask ChatGPT the way a real customer would and show whether you appear.</p></div>'
        '<div class="full-feature"><span>02</span><strong>Priority fix list</strong><p>See what to fix first, in plain English, so you do not waste time.</p></div>'
        '<div class="full-feature"><span>03</span><strong>Ready-to-paste fixes</strong><p>Get schema and page-by-page details you or your web person can use.</p></div>'
        '</div>'
        '</div>'
        '<div class="full-picture-proof">'
        '<div class="proof-card-top">'
        '<div>'
        '<div class="proof-label">Example included</div>'
        '<h3>See what the PDF looks like</h3>'
        '</div>'
        '<div class="proof-price">$10</div>'
        '</div>'
        '<p>Preview a sample report first, then download the full report generated from this scan.</p>'
        '<div class="proof-mini-list">'
        '<div><span></span>AI recommendation snapshot</div>'
        '<div><span></span>Detailed findings</div>'
        '<div><span></span>Specific code/schema fixes</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    # Scroll target for the mid-page "Want the exact fixes?" CTA above.
    st.markdown('<div id="full-report"></div>', unsafe_allow_html=True)
    st.markdown(full_report_html, unsafe_allow_html=True)

    # Owner sees the example alongside the real download; visitors see the
    # example and then the purchase form (details → checkout).
    if is_admin():
        btn_left, btn_right = st.columns(2, gap="medium")
        with btn_left:
            st.download_button(
                "EXAMPLE — see sample PDF ↗",
                data=build_fake_example_pdf(),
                file_name="localagentready_example_report.pdf",
                mime="application/pdf",
                key="example_pdf_link",
                use_container_width=True,
            )
        with btn_right:
            render_full_report_preview(report)
    else:
        st.download_button(
            "EXAMPLE — see sample PDF ↗",
            data=build_fake_example_pdf(),
            file_name="localagentready_example_report.pdf",
            mime="application/pdf",
            key="example_pdf_link",
            use_container_width=True,
        )
        render_report_purchase(report)


def render_admin_diagnostics() -> None:
    """Owner-only setup checks, shown when the app is opened with
    ?admin=<ADMIN_TOKEN>. Lets you confirm the Google Sheets store is wired up
    — with a real probe write — without having to run a full scan or submit the
    lead form first."""
    with st.expander("⚙️ Admin diagnostics (owner only)", expanded=False):
        store_ok = datastore.is_available()
        st.markdown(
            f"- **Data store (Google Sheets):** {'✅ configured' if store_ok else '❌ not configured'}\n"
            f"- **Payment link (Stripe):** {'✅ set' if _secret('STRIPE_PAYMENT_LINK') else '❌ not set'}\n"
            f"- **Current typical score:** {current_typical_score()}/100"
        )

        # When the store is off, show exactly which condition fails so setup
        # can be fixed without guessing.
        if not store_ok:
            d = datastore.availability_detail()
            st.markdown(
                f"  - Library installed (gspread/google-auth): "
                f"{'✅' if d.get('library') else '❌ — run pip install -r requirements.txt, and on Streamlit Cloud reboot the app so new deps install'}\n"
                f"  - `SHEET_ID` secret present: {'✅' if d.get('sheet_id') else '❌'}\n"
                f"  - `[gcp_service_account]` secret present: {'✅' if d.get('service_account') else '❌'}"
            )
            if d.get("library_error"):
                st.caption(f"Import error: {d['library_error']}")
            if d.get("secrets_error"):
                st.caption(f"Secrets error (often a TOML parse problem in the key): {d['secrets_error']}")
        if datastore.availability_detail().get("service_account"):
            st.caption(f"Service account (share the sheet with this as Editor): {datastore.service_account_email()}")

        if st.button("Test data store", key="admin_test_store", use_container_width=True):
            ok, msg = datastore.diagnose_write()
            current_typical_score.clear()
            if ok:
                st.success(f"{msg} Typical score now reads {datastore.get_typical_score()}/100.")
            else:
                st.error(msg)


# ---------------- Styling ----------------

st.markdown("""
<style>
    :root {
        --ink: #0b0b0f;
        --muted: #5f6368;
        --line: #e8e3dc;
        --soft: #f7f2ea;
        --soft-2: #fbfaf7;
        --accent: #E76F51;
        --accent-dark: #D85F43;
        /* Warm, on-brand tiers instead of stock SaaS blue/green/red — same
           good→bad ordering, communicated through contrast and warmth
           (ink → accent → amber → muted taupe) rather than traffic-light hues. */

          /* Strong / best - premium black */
  --tier-strong-bg: #F2F1EF;
  --tier-strong-text: #111114;
  --tier-strong-border: #111114;

  /* Good - modern muted green */
  --tier-good-bg: #EEF8F1;
  --tier-good-text: #1F6B43;
  --tier-good-border: #75B88A;

  /* Medium - warm amber */
  --tier-medium-bg: #FFF8E8;
  --tier-medium-text: #5C3A00;
  --tier-medium-border: #B87900;

  /* Low / needs work - coral red */
  --tier-low-bg: #FFF3EE;
  --tier-low-text: #7A2416;
  --tier-low-border: #D96A4D;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(231,111,81,0.10), transparent 32%),
            radial-gradient(circle at 85% 8%, rgba(0,0,0,0.045), transparent 28%),
            #fffdf9;
        color: var(--ink);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.05rem;
        padding-bottom: 3rem;
    }

    header, footer, #MainMenu { visibility: hidden; }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 0.25rem auto 1.45rem auto;
        color: var(--ink);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-weight: 850;
        letter-spacing: -0.03em;
        font-size: 1.15rem;
    }

    .brand-mark {
        width: 45px;
        height: 45px;
        border-radius: 12px;
        background: linear-gradient(135deg, #0b0b0f 0%, #24242a 100%);
        color: white;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.92rem;
        letter-spacing: -0.04em;
        // box-shadow: 0 10px 26px rgba(0,0,0,0.14);
        position: relative;
    }

    .brand-mark:after {
        content: "";
        position: absolute;
        right: 4px;
        bottom: 4px;
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: var(--accent);
    }

    .brand-name {
      font-size: 1.4rem;
      font-weight: 600;
    }

.top-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.55rem 0.85rem;
  border-radius: 999px;
  background: transparent;
  border: 1px solid rgba(8, 8, 10, 0.10);
  color: var(--ink);
  font-size: 1rem;
  font-weight: 500;
  box-shadow: none;
}

.st-key-back_button .stButton button {
 height: 25px !important;
 min-height: 25px !important;
 background: transparent !important;
 color: black !important;
}

.st-key-back_button .stButton button:hover {
 background: transparent !important;
 color: var(--accent) !important;
}

.top-pill:before {
    content: "";
    position: relative;
    # right: 2px;
    # bottom: 20px;
    width: 12px;
    height: 12px;
    border-radius: 999px;
    background: var(--accent);
}

    .hero-wrap {
        text-align: center;
        padding: 1.7rem 0 1rem;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.75rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        color: #3b3b3f;
        background: rgba(255,255,255,0.72);
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    .hero-title {
        max-width: 980px;
        margin: 0 auto;
        font-size: clamp(2.8rem, 5.6vw, 4.85rem);
        line-height: 0.97;
        font-weight: 500;
        letter-spacing: -0.072em;
        color: var(--ink);
    }

    .hero-title .soft-highlight {
        display: inline-block;
        position: relative;
        z-index: 1;
    }

    .hero-title .soft-highlight::after {
      content: "";
      position: absolute;
      left: 0.04em;
      right: 0.04em;
      bottom: -0.1em;
      height: 0.10em;
      background: #f3b8a8;
      border-radius: 999px;
      z-index: -1;
    }

    .hero-subtitle {
        max-width: 790px;
        margin: 1.25rem auto 1.85rem auto;
        font-size: clamp(1.08rem, 2.1vw, 1.32rem);
        line-height: 1.48;
        color: #252529;
        font-weight: 420;
    }

    .scan-row-label {
        max-width: 1040px;
        margin: 0 auto 0.55rem auto;
        color: #6c6862;
        font-size: 0.95rem;
        text-align: left;
    }

    div[data-baseweb="input"] {
        height: 68px !important;
        min-height: 68px !important;
        background: transparent !important;
        box-shadow: none !important;
        display: flex !important;
        align-items: center !important;
        overflow: visible !important;
    }

    div[data-baseweb="input"] input {
        height: 64px !important;
        min-height: 64px !important;
        line-height: 64px !important;
        font-size: 1.18rem !important;
        color: var(--ink) !important;
        background: rgba(255,255,255,0.92) !important;
        padding: 0 1.25rem !important;
        border: 1.5px solid #ded8cf !important;
        border-radius: 0.5rem !important;
        // box-shadow: 0 18px 50px rgba(12,12,15,0.08) !important;
    }

    div[data-baseweb="input"] input:focus {
        border-color: var(--ink) !important;
        // box-shadow: 0 0 0 4px rgba(231,111,81,0.14) !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #9a9894 !important;
        opacity: 1 !important;
    }

    div[data-testid="stTextInput"] { margin-bottom: 0 !important; }
    div[data-testid="stTextInput"] label { display: none !important; }

    div[data-testid="stButton"] button {
        height: 64px !important;
        min-height: 64px !important;
        width: 200px !important;
        background: var(--ink) !important;
        color: white !important;
        font-weight: 820 !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-size: 1.04rem !important;
        margin-top: 0 !important;
        // box-shadow: 0 18px 42px rgba(0,0,0,0.20) !important;
        transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
    }

    div[data-testid="stButton"] button:hover {
        background: var(--accent) !important;
        color: white !important;
        transform: translateY(-1px);
        // box-shadow: 0 18px 34px rgba(231,111,81,0.25) !important;
    }

    div[data-testid="stButton"] p {
        font-weight: 560 !important;
        font-size: 1.13rem !important;
    }

    .mini-proof {
        display: flex;
        justify-content: center;
        gap: 0.55rem;
        flex-wrap: wrap;
        color: #aaaaaa;
        font-size: 1rem;
        # margin-top: 1.3rem;
        # margin-bottom: 1.4rem;
    }

    .mini-proof span {
        display: inline-flex;
        align-items: center;
        padding: 0.45rem 0.72rem;
        # background: rgba(255,255,255,0.70);
        # border: 1px solid var(--line);
        border-radius: 999px;
    }

    .example-wrap {
        max-width: 640px;
        margin: 0 auto 2.3rem auto;
        text-align: center;
    }

    .example-label {
        color: #8a857e;
        font-size: 0.9rem;
        margin-bottom: 0.55rem;
        letter-spacing: 0.02em;
    }

    .example-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        background: rgba(255,255,255);
        border: 1px dashed #d8d2c8;
        border-radius: 20px;
        padding: 1rem 1.3rem;
        text-align: left;
    }

    .example-domain {
        font-weight: 760;
        color: var(--ink);
        font-size: 1rem;
    }

    .example-issue {
        color: #6c6862;
        font-size: 0.92rem;
        margin-top: 0.15rem;
    }

    .example-score-badge {
        flex: 0 0 auto;
        text-align: center;
        background: var(--soft-2);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.55rem 0.95rem;
    }

    .example-score-badge .num {
        font-size: 1.4rem;
        font-weight: 900;
        color: var(--ink);
        line-height: 1;
    }

    .example-score-badge .band {
        font-size: 0.72rem;
        color: var(--tier-medium-text);
        font-weight: 760;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .section-band {
        margin-top: 2.4rem;
        padding: 1.1rem 0 0.6rem;
    }

    .result-card, .modern-card {
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--line);
        border-radius: 0.5rem;
        padding: 1.55rem;
        min-height: 260px;
        // box-shadow: 0 18px 55px rgba(12,12,15,0.07);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    .stButton {
    text-align: center;
    }
    .card-title {
        font-size: 1.08rem;
        font-weight: 850;
        color: var(--ink);
        margin-bottom: 0.65rem;
    }

    .card-underline {
        width: 58px;
        height: 6px;
        background: var(--accent);
        border-radius: 999px;
        margin-bottom: 1.25rem;
    }

    .small-copy h3 {
        margin: 0 0 0.75rem 0;
        font-size: clamp(1.35rem, 2.2vw, 1.8rem);
        line-height: 1.15;
        font-weight: 850;
        letter-spacing: -0.045em;
        color: #1f1f23;
    }

    .small-copy p {
        color: #313238;
        font-size: 1.02rem;
        line-height: 1.55;
        margin: 0;
    }

    .moments-wrap {
        margin: 1.4rem 0 1.4rem 0;
        padding: 2rem;
        border-radius: 0.5rem;
        background: #0b0b0f;
        color: white;
        // box-shadow: 0 22px 70px rgba(0,0,0,0.18);
    }

    .moments-wrap h2 {
        margin: 0 0 0.55rem 0;
        font-size: clamp(1.8rem, 3vw, 2.6rem);
        line-height: 1.05;
        font-weight: 400;
        letter-spacing: -0.055em;
    }

    .moments-wrap p {
        max-width: 760px;
        color: rgba(255,255,255,0.78);
        font-size: 1.06rem;
        line-height: 1.55;
        margin: 0 0 1.25rem 0;
    }

    .moment-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .moment-card {
        # border: 1px solid rgba(255,255,255,0.14);
        # background: rgba(255,255,255,0.2);
        background: var(--soft-2);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem;
    }

    .moment-card strong {
        display: block;
        margin-bottom: 0.35rem;
        color: var(--ink);
    }

    .moment-card span {
        display: block;
        color: var(--ink);
        line-height: 1.45;
        font-size: 0.96rem;
    }

    .metric-box {
        background: var(--soft-2);
        border: 1px solid var(--line);
        border-radius: 0.5rem;
        padding: 1.25rem 1.35rem;
        text-align: left;
        height: 100%;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: clamp(2.2rem, 5vw, 4rem);
        line-height: 0.95;
        font-weight: 900;
        letter-spacing: -0.065em;
        color: var(--ink);
    }

    .status-pill {
        display: inline-flex;
        margin-top: 0.65rem;
        padding: 0.42rem 0.72rem;
        border-radius: 999px;
        font-weight: 760;
        font-size: 0.92rem;
    }

    .status-strong { background: var(--tier-strong-bg); color: var(--tier-strong-text); }
    .status-good { background: var(--tier-good-bg); color: var(--tier-good-text); }
    .status-medium { background: var(--tier-medium-bg); color: var(--tier-medium-text); }
    .status-low { background: var(--tier-low-bg); color: var(--tier-low-text); }

    .message-card {
        margin-top: 1rem;
        background: #0b0b0f;
        color: white;
        border-radius: 0.5rem;
        padding: 1.15rem 1.25rem;
        line-height: 1.5;
    }

    .benchmark-card {
        margin-top: 0.85rem;
        background: var(--soft-2);
        border: 1px solid var(--line);
        border-radius: 0.5rem;
        padding: 1.1rem 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .benchmark-copy {
        color: #33333a;
        font-size: 1.2rem;
        line-height: 1.45;
    }

    .benchmark-copy .fine-print {
        display: block;
        color: var(--ink);
        font-size: 1rem;
        margin-top: 0.25rem;
    }

    .benchmark-delta {
        flex: 0 0 auto;
        font-weight: 850;
        font-size: 1.05rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
    }

    .benchmark-ahead { color: var(--tier-good-text); }
    .benchmark-behind { color: var(--tier-low-text); }

    /* Benchmark shown directly under the score gauge — centered and width-capped
       so it reads as part of the headline block, not a full-width banner. */
    .benchmark-centered {
        max-width: 820px;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 1.8rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 870;
        color: var(--ink);
        letter-spacing: -0.035em;
        margin: 1.4rem 0 0.9rem 0;
    }

    .fix-item {
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        padding: 0.95rem 1rem;
        border-radius: 16px;
        margin-bottom: 0.72rem;
        border: 1px solid var(--line);
        color: #222222;
        line-height: 1.45;
    }

    .fix-dot {
        flex: 0 0 auto;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 999px;
        font-weight: 820;
        font-size: 0.88rem;
        background: rgba(255,255,255,0.8);
        border: 1px solid rgba(0,0,0,0.07);
    }


    .fix-number {
        flex: 0 0 auto;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        margin-top:5px;
        # border-radius: 999px;
        font-weight: 820;
        font-size: 3rem;
        # background: rgba(255,255,255,0.8);
        # border: 1px solid rgba(0,0,0,0.07);
    }

    .fix-dot {
        width: 10px;
        height: 10px;
        margin-top: 0.45rem;
        background: currentColor;
        border: none;
    }

    .fix-copy { flex: 1; }
    .fix-high .fix-number {
        color: var(--tier-low-text);
    }
    .fix-low, .fix-high, .fix-medium { border-color: #ddd; color: #1f1f23; }
    .fix-medium .fix-number {
        color: var(--tier-medium-text);
    }
    .fix-low .fix-number {
        color: var(--tier-good-text);
    }

    .signal-card {
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--line);
        border-radius: 0.5rem;
        padding: 1.25rem;
        // box-shadow: 0 18px 55px rgba(12,12,15,0.05);
    }

    .signal-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 0;
        border-bottom: 1px solid var(--line);
        gap: 1rem;
    }

    .signal-row:last-child { border-bottom: none; }
    .signal-row span { color: #33333a; }
    .signal-row strong { white-space: nowrap; }
    .signal-good { color: var(--tier-good-text); }
    .signal-medium { color: var(--tier-medium-text); }
    .signal-low { color: var(--tier-low-text); }
    .signal-neutral { color: var(--ink); }

    .compare-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
    }

    .compare-table th, .compare-table td {
        padding: 0.55rem 0.7rem;
        text-align: center;
        border-bottom: 1px solid var(--line);
    }

    .compare-table th:first-child, .compare-table td:first-child {
        text-align: left;
        color: var(--muted);
        font-weight: 500;
    }

    .compare-table thead th {
        font-weight: 760;
        color: var(--ink);
    }

    .compare-score-row td {
        font-weight: 800;
        font-size: 1.05rem;
    }

    .compare-table .cmp-yes { color: var(--tier-good-text); font-weight: 800; }
    .compare-table .cmp-no { color: var(--tier-low-text); }

    .lead-card {
        margin-top: 1.8rem;
        padding: 1.55rem;
        border-radius: 0.5rem 0.5rem 0 0;
        background: #0b0b0f;
        color: white;
        // box-shadow: 0 22px 70px rgba(0,0,0,0.16);
    }

    .lead-card h3 {
        margin: 0 0 0.25rem 0;
        font-size: 1.65rem;
        line-height: 1.15;
        letter-spacing: -0.045em;
    }

    .lead-card p {
        margin: 0 0 0.5rem 0;
        color: rgba(255,255,255,0.72);
    }

    .lead-price-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.90);
        border: 1px solid rgba(255,255,255,0.18);
        color: black;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .lead-card-grid {
        display: flex;
        align-items: center;
        gap: 1.6rem;
        flex-wrap: wrap;
    }

    .lead-card-copy {
        flex: 1 1 260px;
    }

    /* The example proof card reuses .example-* styling but sits on the
       lead-card's dark background here, so label/text colors are tuned
       for contrast instead of the light-page versions above. */
    .lead-card .example-wrap {
        flex: 1 1 300px;
        max-width: 380px;
        margin: 0;
        text-align: left;
    }

    .lead-card .example-label {
        color: rgba(255,255,255,0.72);
        text-decoration-color: rgba(255,255,255,0.35);
    }

    div[data-testid="stDownloadButton"] button {
        border-radius: 0 0 0.5rem 0.5rem !important;
    }
    /* Report-purchase submit — matches the accent "Compare" button. */
    div[data-testid="stFormSubmitButton"] button {
        background: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 820 !important;
        height: 58px !important;
        min-height: 58px !important;
        font-size: 1.03rem !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--accent-dark) !important;
        color: #ffffff !important;
        transform: translateY(-1px);
    }

    /* The "EXAMPLE — see the full PDF report" button (a real
       st.download_button — see the comment at its call site for why a
       raw <a href="data:..."> link can't be used instead) sits directly
       under the example card, styled as a solid black pill so it reads
       as one attached unit rather than a floating secondary control.
       It's rendered as its own full-width Streamlit element (not inside
       a column, since Streamlit's column widths are based on the full
       page width and don't line up with the card's internal flex sizing)
       — width/alignment here are matched by hand to the example card's
       own max-width (380px) and the lead-card's padding (1.55rem) instead. */
    .st-key-example_pdf_link {
        /* width (not just max-width) is required: Streamlit's wrapper is a
           flex item that otherwise shrinks to its button's content size
           regardless of max-width. */
        # width: 380px;
        # max-width: 380px;
        flex: none;
        # margin: -0.85rem 1.55rem 2.3rem auto;
    }

    .st-key-example_pdf_link button {
        width: 100%;
        background: var(--ink) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0 !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        padding: 0.65rem 1rem !important;
        height: auto !important;
        min-height: auto !important;
    }

    .st-key-example_pdf_link button:hover {
        background: var(--accent) !important;
        color: #ffffff !important;
        transform: none !important;
        box-shadow: none !important;
    }

    .stAlert {
        border-radius: 16px;
    }

    @media (max-width: 900px) {
        .topbar { margin-bottom: 1rem; }
        .top-pill { display: none; }
        .hero-wrap { padding-top: 1.3rem; }
        .input-shell { padding: 0.45rem; }
        .moment-grid { grid-template-columns: 1fr; }
        .mini-proof { margin-bottom: 1.5rem; }
        .example-card { flex-direction: column; align-items: flex-start; }
        .benchmark-card { flex-direction: column; align-items: flex-start; }
        .st-key-example_pdf_link { width: 100% !important; margin: -0.85rem 0 2.3rem 0 !important; }
        div[data-baseweb="input"], div[data-testid="stButton"] button {
            height: 58px !important;
            min-height: 58px !important;
        }
        div[data-baseweb="input"] input {
            height: 56px !important;
            min-height: 56px !important;
            line-height: 56px !important;
        }
        .category-grid { grid-template-columns: 1fr !important; }
    }

    /* ---- Results page ---- */
    .results-eyebrow {
        text-align: center;
        color: var(--muted);
        font-size: 1rem;
        margin: 0.4rem 0 0.1rem 0;
    }

    .results-site {
        text-align: center;
        font-size: clamp(1.9rem, 4vw, 3rem);
        font-weight: 850;
        letter-spacing: -0.04em;
        color: var(--ink);
        margin: 0 0 1.2rem 0;
        word-break: break-word;
    }

    /* ---- Results score intro + two decision cards ---- */
    /* ---- Strong two-panel axis section (results top) ---- */
    .axis-panel-wrap {
        max-width: 1120px;
        margin: 0.35rem auto 2.25rem auto;
    }

    .axis-panel-header {
        text-align: center;
        color: var(--ink);
        font-size: clamp(1.5rem, 2.8vw, 2.15rem);
        line-height: 1.05;
        letter-spacing: -0.055em;
        margin: 0 0 1.45rem 0;
        font-weight: 900;
    }

    .axis-panel-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1.05rem;
    }

    .axis-panel {
        position: relative;
        overflow: hidden;
        border-radius: 0.9rem;
        border: 1px solid var(--line);
        padding: 1.55rem 1.65rem;
        min-height: 375px;
        display: flex;
        flex-direction: column;
        box-shadow: 0 18px 52px rgba(12,12,15,0.055);
    }

    .axis-panel-free {
        background:
            radial-gradient(circle at bottom right, rgba(231,111,81,0.19), transparent 34%),
            radial-gradient(circle at top left, rgba(255,255,255,0.70), transparent 30%),
            #FFF1EA;
        border-color: #E9B9A8;
    }

    .axis-panel-free::after {
        content: "";
        position: absolute;
        right: -80px;
        bottom: -90px;
        width: 310px;
        height: 310px;
        background-image: radial-gradient(rgba(231,111,81,0.22) 1.5px, transparent 1.5px);
        background-size: 13px 13px;
        opacity: 0.55;
        pointer-events: none;
    }

    .axis-panel-paid {
        background:
            radial-gradient(circle at top right, rgba(231,111,81,0.12), transparent 30%),
            radial-gradient(circle at bottom left, rgba(255,255,255,0.62), transparent 34%),
            #F5EDE3;
        border-color: #E2D3C4;
    }

    .axis-panel-top {
        position: relative;
        z-index: 1;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.35rem;
        flex-wrap: wrap;
    }

    .axis-panel-kicker-wrap {
        display: inline-flex;
        align-items: center;
        gap: 0.65rem;
    }

    .axis-panel-number {
        width: 34px;
        height: 34px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 920;
        font-size: 0.92rem;
    }

    .axis-panel-number-free {
        background: var(--accent);
        color: #ffffff;
        box-shadow: 0 8px 20px rgba(231,111,81,0.22);
    }

    .axis-panel-number-paid {
        background: #6B3F00;
        color: #ffffff;
        box-shadow: 0 8px 20px rgba(107,63,0,0.16);
    }

    .axis-panel-kicker {
        color: #5F5A55;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-size: 0.84rem;
        font-weight: 920;
    }

    .axis-panel-status,
    .axis-panel-lock {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 0.55rem 0.95rem;
        font-size: 0.88rem;
        font-weight: 900;
        white-space: nowrap;
    }

    .axis-panel-status {
        background: #ffffff;
        border: 1px solid #F0C8BA;
        color: #8A4B00;
        box-shadow: 0 8px 18px rgba(12,12,15,0.035);
    }

    .axis-panel-lock {
        background: var(--ink);
        color: #ffffff;
        box-shadow: 0 10px 24px rgba(11,11,15,0.12);
    }

    .axis-panel-main {
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 1rem;
        align-items: center;
        margin-bottom: 1.35rem;
    }

    .axis-panel-icon {
        width: 76px;
        height: 76px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 2rem;
    }

    .axis-panel-icon-free {
        background: #FFE1D6;
        color: var(--accent);
        border: 1px solid #F0BBA9;
    }

    .axis-panel-icon-paid {
        background: #F5DEC6;
        color: #6B3F00;
        border: 1px solid #E7C8A5;
    }

    .axis-panel h3 {
        color: var(--ink);
        font-size: clamp(1.45rem, 2.3vw, 1.95rem);
        line-height: 1.05;
        letter-spacing: -0.055em;
        margin: 0 0 0.55rem 0;
        font-weight: 920;
    }

    .axis-panel p {
        color: #2F2B27;
        font-size: 1.05rem;
        line-height: 1.48;
        margin: 0;
    }

    .axis-score-row {
        position: relative;
        z-index: 1;
        margin-top: auto;
        display: grid;
        grid-template-columns: auto 1px 1fr;
        gap: 1.25rem;
        align-items: end;
    }

    .axis-score {
        display: flex;
        align-items: baseline;
        gap: 0.34rem;
        line-height: 1;
    }

    .axis-score-num {
        font-size: clamp(5.1rem, 8.5vw, 7.4rem);
        font-weight: 930;
        letter-spacing: -0.09em;
        color: #8A4B00;
    }

    .axis-score-den {
        color: #4F4B46;
        font-size: 1.65rem;
        font-weight: 760;
    }

    .axis-score-divider {
        width: 1px;
        height: 84px;
        background: #E9B9A8;
        align-self: center;
    }

    .axis-score-copy {
        color: #2F2B27;
        font-size: 1.03rem;
        line-height: 1.5;
        max-width: 22rem;
        padding-bottom: 0.3rem;
    }

    .axis-panel-list {
        position: relative;
        z-index: 1;
        list-style: none;
        margin: 0 0 1.35rem 0;
        padding: 0;
        color: #2F2B27;
        font-size: 1.03rem;
        line-height: 1.5;
    }

    .axis-panel-list li {
        position: relative;
        margin-bottom: 0.62rem;
        padding-left: 2rem;
    }

    .axis-panel-list li::before {
        content: "✓";
        position: absolute;
        left: 0;
        top: 0.03rem;
        width: 1.25rem;
        height: 1.25rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #8A4B00;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 900;
    }

    .axis-panel-action {
        position: relative;
        z-index: 1;
        margin-top: auto;
        border-top: 1px solid #E2D3C4;
        padding-top: 1.05rem;
    }

    .axis-panel-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        min-height: 58px;
        padding: 0.95rem 1.25rem;
        border-radius: 0.75rem;
        background: var(--ink);
        color: #ffffff !important;
        text-decoration: none !important;
        font-size: 1.05rem;
        font-weight: 900;
        letter-spacing: -0.025em;
        box-shadow: 0 14px 26px rgba(11,11,15,0.16);
        transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
    }

    .axis-panel-button:hover {
        transform: translateY(-1px);
        background: var(--accent);
        box-shadow: 0 18px 30px rgba(231,111,81,0.22);
        color: #ffffff !important;
    }

    .axis-panel-note {
        margin-top: 0.75rem;
        color: #5F5A55;
        text-align: center;
        font-size: 0.92rem;
        line-height: 1.4;
    }

    @media (max-width: 900px) {
        .axis-panel-grid {
            grid-template-columns: 1fr;
        }

        .axis-panel {
            min-height: auto;
            padding: 1.35rem;
        }

        .axis-panel-main {
            grid-template-columns: 1fr;
        }

        .axis-score-row {
            grid-template-columns: 1fr;
            gap: 0.6rem;
        }

        .axis-score-divider {
            width: 100%;
            height: 1px;
        }
    }

    .discovery-preview {
        color: #33333a;
        font-size: 1rem;
        margin: 0.5rem 0 1rem 0;
    }
    .discovery-preview strong { color: var(--ink); }

    .locked-card {
        border: 1px dashed #d8d2c8;
        border-radius: 0.5rem;
        background: rgba(255,255,255,0.55);
        padding: 1.3rem 1.5rem;
    }
    .locked-head {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.7rem;
    }
    .locked-lock { font-size: 1.2rem; }
    .locked-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.32rem 0.7rem;
        border-radius: 999px;
        background: var(--ink);
        color: #fff;
        font-size: 0.85rem;
        font-weight: 720;
    }
    .locked-card p { color: #33333a; line-height: 1.55; margin: 0 0 0.5rem 0; }
    .locked-card .locked-sub { color: var(--muted); font-size: 0.95rem; margin: 0; }

    .exec-summary {
        max-width: 820px;
        margin: 0 auto 2rem auto;
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--line);
        border-radius: 0.5rem;
        padding: 1.4rem 1.6rem;
    }
    .exec-summary .exec-label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.78rem;
        color: var(--muted);
        font-weight: 760;
        margin-bottom: 0.7rem;
    }
    .exec-summary ul { margin: 0; padding-left: 1.1rem; }
    .exec-summary li { color: #313238; line-height: 1.55; margin-bottom: 0.55rem; }
    .exec-summary li:last-child { margin-bottom: 0; }

    .category-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1.6rem;
    }
    .category-card {
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
        border-radius: 0.5rem;
        padding: 1.3rem 1.4rem;
        display: flex;
        flex-direction: column;
    }
    .cat-head { display: flex; align-items: flex-start; gap: 0.8rem; margin-bottom: 0.8rem; }
    .cat-icon {
        flex: 0 0 auto;
        width: 40px; height: 40px;
        border-radius: 10px;
        background: var(--soft);
        border: 1px solid var(--line);
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 1.15rem;
    }
    .cat-titles { flex: 1; }
    .cat-title { font-weight: 820; color: var(--ink); font-size: 1.05rem; letter-spacing: -0.02em; }
    .cat-status { font-size: 0.85rem; font-weight: 720; margin-top: 0.1rem; }
    .cat-score { font-size: 1.7rem; font-weight: 900; letter-spacing: -0.03em; }
    .cat-status-good { color: var(--tier-good-text); }
    .cat-status-strong { color: var(--tier-strong-text); }
    .cat-status-medium { color: var(--tier-medium-text); }
    .cat-status-low { color: var(--tier-low-text); }
    .cat-desc { color: #33333a; font-size: 0.96rem; line-height: 1.5; flex: 1; margin-bottom: 1rem; }
    .cat-bar {
        height: 8px;
        background: var(--line);
        border-radius: 999px;
        overflow: hidden;
    }
    .cat-bar-fill { height: 100%; border-radius: 999px; }
    .cat-fill-good { background: var(--accent); }
    .cat-fill-medium { background: var(--tier-medium-text); }
    .cat-fill-low { background: var(--tier-low-text); }

/* ---- Premium Overall Summary ---- */
.summary-panel {
  background:
    radial-gradient(circle at top right, rgba(231,111,81,0.20), transparent 34%),
    #0B0B0F;
  color: #ffffff;
  border-radius: 0.5rem;
  padding: 2.6rem;
  margin: 2.4rem 0 1.4rem 0;
}

.summary-top {
  display: flex;
  justify-content: space-between;
  gap: 2rem;
  align-items: flex-start;
  margin-bottom: 2rem;
}

.summary-eyebrow {
  color: var(--accent);
  font-size: 0.88rem;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 1rem;
}

.summary-panel h2 {
  font-size: clamp(2.4rem, 5vw, 4.2rem);
  line-height: 0.95;
  letter-spacing: -0.065em;
  margin: 0 0 1rem 0;
  color: #ffffff;
}

.summary-subtitle {
  color: #D8D2CC;
  font-size: 1.08rem;
  line-height: 1.55;
  margin: 0;
  max-width: 820px;
}

.summary-score-pill {
  border: 1px solid rgba(255,255,255,0.18);
  border-radius: 999px;
  padding: 0.75rem 1.05rem;
  font-weight: 900;
  white-space: nowrap;
  background: rgba(255,255,255);
}

.summary-insight-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-top: 2rem;
}

.summary-insight-card {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 0.5rem;
  padding: 1.15rem;
}

.summary-insight-kicker {
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.75rem;
}

.summary-insight-card p {
  color: #F7F1EA;
  font-size: 0.98rem;
  line-height: 1.55;
  margin: 0;
}

.summary-benchmark {
  margin-top: 1.15rem;
  background: #ffffff;
  color: #111114;
  border-radius: 0.5rem;
  padding: 1.05rem 1.2rem;
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: center;
}

.summary-benchmark strong {
  display: block;
  margin-bottom: 0.25rem;
}

.summary-benchmark span {
  color: #5F5A55;
}

/* ---- Premium Moments Section ---- */
.moments-wrap {
  background:
    radial-gradient(circle at bottom right, rgba(231,111,81,0.20), transparent 36%),
    #0B0B0F !important;
  border-radius: 0.5rem !important;
  padding: 2.4rem !important;
  margin: 2rem 0 !important;
}

.moments-eyebrow {
  color: var(--accent);
  font-size: 0.85rem;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 0.9rem;
}

.moments-wrap h2 {
  color: #ffffff !important;
  font-size: clamp(2rem, 4vw, 3.2rem) !important;
  font-weight: 850 !important;
  letter-spacing: -0.06em !important;
}

.moment-card {
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  color: #ffffff !important;
}

.moment-card strong {
  color: #ffffff !important;
}

.moment-card span {
  color: #D8D2CC !important;
}

.moment-number {
  display: inline-flex;
  width: 34px;
  height: 34px;
  border-radius: 999px;
  align-items: center;
  justify-content: center;
  background: rgba(231,111,81,0.16);
  color: var(--accent);
  font-weight: 900;
  margin-bottom: 0.75rem;
}

/* ---- Premium Competitor Feature ---- */
.competitor-feature {
  background:
    radial-gradient(circle at bottom right, rgba(231,111,81,0.22), transparent 34%),
    #d0673d;
  color: #ffffff;
  border-radius: 0.5rem;
  padding: 2.6rem;
  margin: 2.4rem 0 1rem 0;
}

.competitor-eyebrow {
  color: #0B0B0F;
  font-size: 0.85rem;
  font-weight: 900;
  text-transform: uppercase;
  letter-spacing: 0.11em;
  margin-bottom: 0.8rem;
}

.competitor-feature h2 {
  margin: 0 0 0.7rem 0;
  font-size: clamp(2.1rem, 4vw, 3.4rem);
  line-height: 0.98;
  font-weight: 900;
  letter-spacing: -0.06em;
  color: #ffffff;
}

.competitor-feature p {
  max-width: 780px;
  color: white;
  font-size: 1.08rem;
  line-height: 1.55;
  margin: 0 0 1.5rem 0;
}

.competitor-micro-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-top: 1.2rem;
}

.competitor-micro {
  background: rgba(255,255,255,0.8);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 0.5rem;
  padding: 1rem;
}

.competitor-micro-icon {
  width: 34px;
  height: 34px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #d0673d;
  color: white;
  font-weight: 900;
  font-size: 0.82rem;
  margin-bottom: 0.7rem;
}

.competitor-micro strong {
  display: block;
  color: black;
  margin-bottom: 0.35rem;
}

.competitor-micro span {
  display: block;
  color: black;
  line-height: 1.45;
  font-size: 0.95rem;
}

.competitor-form-intro {
  text-align: center;
  color: #6c6862;
  margin: 0.4rem 0 0.8rem 0;
}

.st-key-run_compare button {
  width: 280px !important;
  background: var(--accent) !important;
  color: #ffffff !important;
}

.st-key-run_compare button:hover {
  background: var(--accent-dark) !important;
  color: #ffffff !important;
}

/* Improve number readability on white cards */
.fix-number {
  opacity: 1 !important;
  font-weight: 900 !important;
}
.fix-high .fix-number { color: var(--tier-low-text) !important; }
.fix-medium .fix-number { color: var(--tier-medium-text) !important; }
.fix-low .fix-number { color: var(--tier-good-text) !important; }

@media (max-width: 900px) {
  .summary-top,
  .summary-benchmark {
    flex-direction: column;
    align-items: flex-start;
  }
  .summary-insight-grid,
  .competitor-micro-grid {
    grid-template-columns: 1fr;
  }
  .summary-panel,
  .competitor-feature,
  .moments-wrap {
    padding: 1.7rem !important;
  }
}


    /* ---- Premium full-report CTA ---- */
    .full-picture-section {
        margin: 2.4rem 0 0.9rem 0;
        padding: 2.6rem;
        border-radius: 0.5rem;
        background:
            radial-gradient(circle at top right, rgba(231,111,81,0.16), transparent 34%),
            linear-gradient(135deg, #0b0b0f 0%, #141418 58%, #241310 100%);
        color: #ffffff;
        display: grid;
        grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.8fr);
        gap: 2rem;
        align-items: stretch;
    }

    .full-picture-eyebrow {
        color: var(--accent);
        font-size: 0.86rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .full-picture-copy h2 {
        margin: 0 0 0.85rem 0;
        color: #ffffff;
        font-size: clamp(2.2rem, 4.4vw, 4rem);
        line-height: 0.96;
        letter-spacing: -0.06em;
        font-weight: 850;
    }

    .full-picture-lede {
        max-width: 760px;
        color: rgba(255,255,255,0.78);
        font-size: 1.12rem;
        line-height: 1.58;
        margin: 0 0 1.35rem 0;
    }

    .full-feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 1.5rem;
    }

    .full-feature {
        background: rgba(255,255,255,0.075);
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 0.5rem;
        padding: 1rem;
    }

    .full-feature span {
        display: inline-flex;
        color: var(--accent);
        font-weight: 900;
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        margin-bottom: 0.55rem;
    }

    .full-feature strong {
        display: block;
        color: #ffffff;
        font-size: 0.98rem;
        line-height: 1.25;
        margin-bottom: 0.45rem;
    }

    .full-feature p {
        margin: 0;
        color: rgba(255,255,255,0.68);
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .full-picture-proof {
        background: rgba(255,255,255,0.96);
        color: var(--ink);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 0.5rem;
        padding: 1.35rem;
        align-self: stretch;
    }

    .proof-card-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 0.9rem;
    }

    .proof-label {
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 900;
        font-size: 0.74rem;
        margin-bottom: 0.35rem;
    }

    .full-picture-proof h3 {
        margin: 0;
        color: var(--ink);
        font-size: 1.35rem;
        line-height: 1.08;
        letter-spacing: -0.04em;
    }

    .proof-price {
        flex: 0 0 auto;
        background: var(--ink);
        color: white;
        border-radius: 999px;
        padding: 0.45rem 0.7rem;
        font-weight: 850;
        font-size: 0.9rem;
    }

    .full-picture-proof p {
        color: #4f4b46;
        font-size: 0.98rem;
        line-height: 1.5;
        margin: 0 0 1rem 0;
    }

    .proof-mini-list {
        display: grid;
        gap: 0.55rem;
    }

    .proof-mini-list div {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: #1f1f23;
        font-weight: 650;
        font-size: 0.94rem;
    }

    .proof-mini-list span {
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: var(--accent);
        flex: 0 0 auto;
    }

    .st-key-example_pdf_link button,
    .st-key-full_report_pdf_link button {
        height: 58px !important;
        min-height: 58px !important;
        width: 100% !important;
        border-radius: 0.5rem !important;
        font-weight: 820 !important;
        font-size: 1.03rem !important;
        border: none !important;
        box-shadow: none !important;
    }

    .st-key-example_pdf_link button {
        background: #ffffff !important;
        color: var(--ink) !important;
        border: 1.5px solid var(--line) !important;
    }

    .st-key-full_report_pdf_link button {
        background: var(--accent) !important;
        color: #ffffff !important;
    }

    .st-key-example_pdf_link button:hover,
    .st-key-full_report_pdf_link button:hover {
        transform: translateY(-1px);
        background: var(--ink) !important;
        color: #ffffff !important;
    }

    .lead-form-heading {
        margin: 2rem 0 0.8rem 0;
        padding: 1.3rem 1.45rem;
        border-radius: 0.5rem;
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--line);
    }

    .lead-form-heading h3 {
        margin: 0 0 0.25rem 0;
        color: var(--ink);
        font-size: 1.35rem;
        line-height: 1.1;
        letter-spacing: -0.04em;
    }

    .lead-form-heading p {
        margin: 0;
        color: var(--muted);
        line-height: 1.45;
    }

    @media (max-width: 900px) {
        .full-picture-section {
            grid-template-columns: 1fr;
            padding: 2rem;
        }

        .full-feature-grid {
            grid-template-columns: 1fr;
        }
    }



    /* ---- Modern purchase form ---- */
    .purchase-panel {
        margin: 1.1rem 0 1rem 0;
        padding: 1.55rem;
        border-radius: 0.5rem;
        background: rgba(255,255,255,0.84);
        border: 1px solid var(--line);
    }

    .purchase-panel-head {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
    }

    .purchase-eyebrow {
        color: var(--accent);
        font-size: 0.8rem;
        font-weight: 900;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .purchase-panel h2 {
        margin: 0 0 0.45rem 0;
        color: var(--ink);
        font-size: clamp(1.75rem, 3vw, 2.5rem);
        line-height: 1;
        letter-spacing: -0.055em;
        font-weight: 850;
    }

    .purchase-panel p {
        margin: 0;
        max-width: 760px;
        color: #4f4b46;
        font-size: 1.02rem;
        line-height: 1.55;
    }

    .purchase-price-badge {
        flex: 0 0 auto;
        background: var(--ink);
        color: #ffffff;
        border-radius: 999px;
        padding: 0.55rem 0.85rem;
        font-weight: 850;
        font-size: 0.95rem;
    }

    .purchase-trust-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.1rem;
    }

    .purchase-trust-row span {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.45rem 0.7rem;
        border-radius: 999px;
        background: #fffdf9;
        border: 1px solid var(--line);
        color: #3f3a35;
        font-weight: 650;
        font-size: 0.9rem;
    }

    .purchase-trust-row span:before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--accent);
    }

    .field-label {
        margin: 0.8rem 0 0.35rem 0;
        color: var(--ink);
        font-size: 0.93rem;
        font-weight: 760;
    }

    .st-key-report_purchase_form {
        margin-top: 0.7rem;
        padding: 1.35rem;
        border-radius: 0.5rem;
        background: rgba(255,255,255,0.72);
        border: 1px solid var(--line);
    }

    .st-key-report_purchase_form div[data-baseweb="input"] input {
        background: #ffffff !important;
        border: 1.5px solid #ded8cf !important;
        border-radius: 0.5rem !important;
    }

    .st-key-report_purchase_form div[data-testid="stFormSubmitButton"] button {
        margin-top: 0.9rem !important;
        background: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.5rem !important;
        height: 60px !important;
        min-height: 60px !important;
        font-weight: 850 !important;
        font-size: 1.05rem !important;
    }

    .st-key-report_purchase_form div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--ink) !important;
        color: #ffffff !important;
        transform: translateY(-1px);
    }

    .form-message {
        margin: 1rem 0 0.75rem 0;
        padding: 1rem 1.15rem;
        border-radius: 0.5rem;
        font-size: 0.98rem;
        line-height: 1.5;
        border: 1px solid var(--line);
    }

    .form-message-success {
        background: #EEF8F1;
        color: #1F6B43;
        border-color: #75B88A;
    }

    .form-message-error {
        background: #FFF3EE;
        color: #7A2416;
        border-color: #D96A4D;
    }

    .checkout-button {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 60px;
        background: var(--ink);
        color: #ffffff !important;
        font-weight: 850;
        font-size: 1.05rem;
        border-radius: 0.5rem;
        text-decoration: none !important;
        margin-top: 0.45rem;
    }

    .checkout-button:hover {
        background: var(--accent);
        color: #ffffff !important;
    }

    /* Shown until all fields are filled — looks inert, not clickable. */
    .checkout-button-disabled,
    .checkout-button-disabled:hover {
        background: var(--soft);
        color: var(--muted) !important;
        border: 1.5px solid var(--line);
        cursor: not-allowed;
    }

    .admin-note {
        margin-top: 0.75rem;
        color: var(--muted);
        font-size: 0.9rem;
    }

    /* Smooth in-page scroll for the "Want the exact fixes?" anchor jump. */
    html { scroll-behavior: smooth; }

    /* ---- Mid-page full-report CTA banner ---- */
    .fix-cta {
        display: flex;
        align-items: center;
        gap: 1.1rem;
        flex-wrap: wrap;
        margin: 0.3rem 0 1.5rem 0;
        padding: 1.15rem 1.35rem;
        border-radius: 0.9rem;
        background: #fff6f1;
        border: 1px solid #f3c9b8;
    }

    .fix-cta-icon {
        flex: 0 0 auto;
        width: 52px;
        height: 52px;
        border-radius: 999px;
        background: #fde7dd;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }

    .fix-cta-copy {
        flex: 1 1 300px;
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }

    .fix-cta-copy strong {
        color: var(--ink);
        font-size: 1.15rem;
        font-weight: 850;
        letter-spacing: -0.02em;
    }

    .fix-cta-copy span {
        color: #5f5a55;
        font-size: 0.98rem;
        line-height: 1.45;
    }

    .fix-cta-button {
        flex: 0 0 auto;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--ink);
        color: #ffffff !important;
        font-weight: 820;
        font-size: 1rem;
        padding: 0.95rem 1.4rem;
        border-radius: 0.6rem;
        text-decoration: none !important;
        white-space: nowrap;
        transition: background 0.12s ease, transform 0.12s ease;
    }

    .fix-cta-button:hover {
        background: var(--accent);
        color: #ffffff !important;
        transform: translateY(-1px);
    }

    @media (max-width: 900px) {
        .fix-cta { flex-direction: column; align-items: flex-start; }
        .fix-cta-button { width: 100%; }
    }

</style>
""", unsafe_allow_html=True)


# ---------------- Header ----------------

st.markdown("""
<div class="topbar">
    <div class="brand"><span class="brand-mark">(B)AI</span><span class="brand-name">Business AI Ready</span></div>
    <div class="top-pill">For local businesses</div>
</div>
""", unsafe_allow_html=True)


# ---------------- Session state ----------------

if "report" not in st.session_state:
    st.session_state["report"] = None
if "competitor_reports" not in st.session_state:
    st.session_state["competitor_reports"] = []
if "view" not in st.session_state:
    st.session_state["view"] = "home"
if "scanning" not in st.session_state:
    st.session_state["scanning"] = False

max_pages = 10
report = st.session_state.get("report")

# A results view with no report to show falls back home (defensive).
if st.session_state["view"] == "results" and not report:
    st.session_state["view"] = "home"

# Owner-only setup/health panel — only rendered with a valid ?admin=<token>.
if is_admin():
    render_admin_diagnostics()


# ---------------- Home view ----------------

if st.session_state["view"] == "home":
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">Is AI finding your business — or your competitors?</div>
        <div class="hero-subtitle">
            People are asking AI for local businesses and services. We check whether your website gives AI enough clear, trustworthy information to recommend you.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="scan-row-label">Enter your website to see how clearly AI can understand your business.</div>', unsafe_allow_html=True)
    col_input, col_button = st.columns([3.4, 1.25], gap="small")
    with col_input:
        url = st.text_input(
            "Website address",
            placeholder="https://yourbusiness.com",
            key="website_url_input",
            on_change=queue_scan_from_input,
        )
    with col_button:
        if st.session_state["scanning"]:
            st.button("Analyzing …", disabled=True, use_container_width=True, key="scan_button_busy")
        else:
            if st.button("Check My Website", use_container_width=True, key="scan_button"):
                if url.strip():
                    st.session_state["pending_url"] = url
                    st.session_state["scanning"] = True
                    st.session_state["scan_error"] = ""
                    st.rerun()
                else:
                    st.error("Please enter a website address.")

    st.markdown(
        '<div class="mini-proof"><span>Free audit — no account required</span><span>Fix what matters first</span></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("scan_error"):
        st.error(f"Scan failed: {st.session_state['scan_error']}")

    # ---- Before-scan marketing cards ----
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""
        <div class="result-card">
            <div class="card-title">What we check</div>
            <div class="card-underline"></div>
            <div class="small-copy">
                <h3>Can AI understand and trust your business?</h3>
                <p>We look for the information AI tools need before they can confidently include your business in recommendations.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="result-card">
            <div class="card-title">Why it matters</div>
            <div class="card-underline"></div>
            <div class="small-copy">
                <h3>Old SEO got you ranked. This gets you recommended.</h3>
                <p>AI Agents now answer people directly — naming the businesses they can clearly read and trust. If AI can't understand your site, it simply recommends someone else.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="result-card">
            <div class="card-title">What you get</div>
            <div class="card-underline"></div>
            <div class="small-copy">
                <h3>Know where you stand and what to fix first.</h3>
                <p>You get a website check, a score out of 100, and a simple action plan showing what to improve first.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
<div class="moments-wrap">
  <div class="moments-eyebrow">Explore. Compare. Decide.</div>
  <h2>Show up in the moments customers are deciding.</h2>
  <p>People use AI to explore options, compare choices, and decide who to contact. Your website needs to make your business easy to understand before that decision happens.</p>
  <div class="moment-grid">
    <div class="moment-card"><div class="moment-number">01</div><strong>When they explore</strong><span>Can AI tell what services you offer?</span></div>
    <div class="moment-card"><div class="moment-number">02</div><strong>When they compare</strong><span>Can AI see why customers should trust you?</span></div>
    <div class="moment-card"><div class="moment-number">03</div><strong>When they act</strong><span>Can they quickly call, book, or request a quote?</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Perform the pending scan. The button above set scanning=True and re-ran,
    # so the disabled "Analyzing …" button is already painted before this
    # blocking crawl runs; when it finishes we switch to the results view.
    if st.session_state["scanning"]:
        try:
            new_report = crawl(
                normalize_url_for_display(st.session_state.get("pending_url", "")),
                max_pages=max_pages,
            )
            st.session_state["report"] = new_report
            st.session_state["competitor_reports"] = []
            st.session_state["view"] = "results"
            # Log this real scan so the "typical score" benchmark adapts over
            # time (best-effort — a failed write never affects the scan). Only
            # the visitor's own scan is logged here, not competitor snapshots.
            datastore.log_scan(new_report.normalized_url, new_report.site_score, new_report.grade)
            current_typical_score.clear()
        except Exception as exc:
            st.session_state["report"] = None
            st.session_state["scan_error"] = str(exc)
        st.session_state["scanning"] = False
        st.rerun()


# ---------------- Results view ----------------

else:
    render_results_page(report, st.session_state.get("competitor_reports", []))
