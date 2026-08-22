#!/usr/bin/env python3
"""PDF generation for the LocalAgentReady full report.

This is the actual $10 deliverable, and it needs to earn that price: not a
reflow of the free on-screen summary, but the technical depth behind it —
a per-page audit (not just one issue per page), a personalized ready-to-paste
schema snippet, an explicit list of exactly which pages lack structured
data, and priority-tagged recommendations — none of which the free scan
shows.
"""

from __future__ import annotations

from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Same warm ink/accent palette as the Streamlit app (app.py's :root tokens),
# so the PDF doesn't feel like a different product from the web report.
INK = colors.HexColor("#0b0b0f")
MUTED = colors.HexColor("#5f6368")
ACCENT = colors.HexColor("#D85F43")
LINE = colors.HexColor("#e3ddd3")
SOFT = colors.HexColor("#f7f2ea")
BODY_TEXT = colors.HexColor("#242428")
HIGH_PRIORITY_HEX = "#9C4A2E"
MEDIUM_PRIORITY_HEX = "#8A6420"
HIGH_PRIORITY_COLOR = colors.HexColor(HIGH_PRIORITY_HEX)

BRAND_STYLE = ParagraphStyle("Brand", fontName="Helvetica-Bold", fontSize=11, textColor=ACCENT, leading=14, spaceAfter=2)
TITLE_STYLE = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=21, textColor=INK, leading=25)
WATERMARK_STYLE = ParagraphStyle("Watermark", fontName="Helvetica-Bold", fontSize=9, textColor=ACCENT, leading=12, spaceBefore=2)
SITE_LABEL_STYLE = ParagraphStyle("SiteLabel", fontName="Helvetica-Bold", fontSize=11.5, textColor=INK, leading=15)
META_STYLE = ParagraphStyle("Meta", fontName="Helvetica", fontSize=9, textColor=MUTED, leading=13)
SCORE_STYLE = ParagraphStyle("Score", fontName="Helvetica-Bold", fontSize=32, textColor=INK, leading=34)
BAND_STYLE = ParagraphStyle("Band", fontName="Helvetica-Bold", fontSize=10.5, textColor=ACCENT, leading=14)
H2_STYLE = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=13, textColor=INK, leading=16, spaceBefore=16, spaceAfter=6)
H3_STYLE = ParagraphStyle("H3", fontName="Helvetica-Bold", fontSize=10, textColor=INK, leading=13, spaceBefore=8, spaceAfter=3)
BODY_STYLE = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, textColor=BODY_TEXT, leading=14.5)
ITEM_STYLE = ParagraphStyle("Item", fontName="Helvetica", fontSize=10, textColor=BODY_TEXT, leading=14.5)
CODE_STYLE = ParagraphStyle("Code", fontName="Courier", fontSize=7.6, textColor=INK, leading=10.5, backColor=SOFT)
TABLE_HEAD_STYLE = ParagraphStyle("TableHead", fontName="Helvetica-Bold", fontSize=8.5, textColor=colors.white, leading=11)
TABLE_CELL_STYLE = ParagraphStyle("TableCell", fontName="Helvetica", fontSize=8.5, textColor=BODY_TEXT, leading=11.5)
PAGE_HEADER_STYLE = ParagraphStyle("PageHeader", fontName="Helvetica-Bold", fontSize=10.5, textColor=INK, leading=14, spaceBefore=10, spaceAfter=2)
PAGE_META_STYLE = ParagraphStyle("PageMeta", fontName="Helvetica", fontSize=8.7, textColor=MUTED, leading=12.5, spaceAfter=3)
PAGE_FLAG_STYLE = ParagraphStyle("PageFlag", fontName="Helvetica", fontSize=8.7, textColor=BODY_TEXT, leading=12.5, spaceAfter=4)
MISSING_SCHEMA_ITEM_STYLE = ParagraphStyle("MissingSchemaItem", fontName="Helvetica-Bold", fontSize=9.5, textColor=HIGH_PRIORITY_COLOR, leading=13)

# Recommendations that touch these areas are the ones actually driving the
# score — schema, structured data, and core NAP fields carry the heaviest
# point weights in the scanner. Everything else is still worth doing, just
# lower-leverage, so it's tagged as a secondary priority instead.
HIGH_PRIORITY_KEYWORDS = (
    "structured data", "json-ld", "schema", "phone number", "address",
)


def _priority_tag(text: str) -> tuple[str, str]:
    lower = text.lower()
    if any(keyword in lower for keyword in HIGH_PRIORITY_KEYWORDS):
        return "HIGH IMPACT", HIGH_PRIORITY_HEX
    return "RECOMMENDED", MEDIUM_PRIORITY_HEX


def _schema_snippet(url: str, phone: str | None) -> str:
    """A ready-to-paste LocalBusiness snippet, personalized with whatever
    real data the scan actually found (URL always; phone when detected) —
    a placeholder-only template gets thrown away, a partially-filled one
    gets used."""
    phone_value = phone or "+1-555-123-4567"
    return f"""<script type="application/ld+json">
{{
  "@context": "https://schema.org",
  "@type": "LocalBusiness",
  "name": "Your Business Name",
  "url": "{url}",
  "telephone": "{phone_value}",
  "address": {{
    "@type": "PostalAddress",
    "streetAddress": "123 Main St",
    "addressLocality": "Your City",
    "addressRegion": "ST",
    "postalCode": "00000",
    "addressCountry": "US"
  }},
  "openingHoursSpecification": [
    {{
      "@type": "OpeningHoursSpecification",
      "dayOfWeek": ["Monday","Tuesday","Wednesday","Thursday","Friday"],
      "opens": "08:00",
      "closes": "18:00"
    }}
  ],
  "sameAs": [
    "https://www.facebook.com/yourbusiness",
    "https://www.google.com/maps/place/your-business"
  ]
}}
</script>"""


def _hr() -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.75, color=LINE, spaceBefore=2, spaceAfter=10)


def _bulleted(items: list[str], ordered: bool) -> ListFlowable:
    flow_items = [ListItem(Paragraph(escape(text), ITEM_STYLE), spaceBefore=4) for text in items]
    kwargs = {"start": 1} if ordered else {}
    return ListFlowable(
        flow_items,
        bulletType="1" if ordered else "bullet",
        leftIndent=14,
        bulletFontSize=9,
        **kwargs,
    )


def _bulleted_recommendations(items: list[str]) -> ListFlowable:
    flow_items = []
    for text in items:
        tag, hex_color = _priority_tag(text)
        markup = f'<font color="{hex_color}"><b>{tag} —</b></font> {escape(text)}'
        flow_items.append(ListItem(Paragraph(markup, ITEM_STYLE), spaceBefore=5))
    return ListFlowable(flow_items, bulletType="1", start=1, leftIndent=14, bulletFontSize=9)


def _kv_table(rows: list[tuple[str, str]]) -> Table:
    data = [
        [Paragraph(f"<b>{escape(k)}</b>", TABLE_CELL_STYLE), Paragraph(escape(v), TABLE_CELL_STYLE)]
        for k, v in rows
    ]
    table = Table(data, colWidths=[2.7 * inch, 3.3 * inch])
    table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
        ("BACKGROUND", (0, 0), (0, -1), SOFT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _ai_discovery_flowables(discovery: dict) -> list:
    """The live "does AI recommend you?" result — the single most compelling
    part of the paid report, since it shows the product's core promise tested
    for real (a live ChatGPT web search), not inferred from page heuristics."""
    flow: list = [Paragraph("Does AI recommend you when customers search?", H2_STYLE), _hr()]

    query = discovery.get("query") or ""
    if query:
        flow.append(Paragraph(f"<b>Simulated customer search:</b> &ldquo;{escape(query)}&rdquo;", BODY_STYLE))
        flow.append(Spacer(1, 4))

    verdict = discovery.get("verdict") or (
        "You appeared in ChatGPT's recommendations." if discovery.get("appears")
        else "You did not appear in ChatGPT's recommendations."
    )
    appears = bool(discovery.get("appears"))
    verdict_hex = "#1F6B43" if appears else HIGH_PRIORITY_HEX
    flow.append(Paragraph(f'<font color="{verdict_hex}"><b>{escape(verdict)}</b></font>', BODY_STYLE))

    recommended = discovery.get("recommended") or []
    if recommended:
        flow.append(Spacer(1, 4))
        flow.append(Paragraph("Who ChatGPT recommended, in order:", H3_STYLE))
        items = []
        for i, biz in enumerate(recommended, start=1):
            name = escape(str(biz.get("name", "")))
            tag = ' <font color="#1F6B43"><b>(this is you)</b></font>' if biz.get("is_you") else ""
            items.append(Paragraph(f"{i}. {name}{tag}", ITEM_STYLE))
        flow.append(ListFlowable(
            [ListItem(p, spaceBefore=2) for p in items], bulletType="bullet", leftIndent=14, bulletFontSize=9
        ))
    flow.append(Spacer(1, 4))
    flow.append(Paragraph(
        "A snapshot for this one query — AI answers vary by wording, location, and over time. Improving the "
        "fixes in this report is how you move from overlooked to recommended.",
        META_STYLE,
    ))
    return flow


def _ai_answer_flowables(ai_answer: dict) -> list:
    """What a real AI assistant said about the business using ONLY its page
    content — surfaces exactly what AI can (and can't) tell a customer today."""
    flow: list = [Paragraph("What an AI assistant says about you", H2_STYLE), _hr()]

    question = ai_answer.get("question") or ""
    if question:
        flow.append(Paragraph(f"<b>Simulated customer question:</b> &ldquo;{escape(question)}&rdquo;", BODY_STYLE))
        flow.append(Spacer(1, 4))

    answer = ai_answer.get("answer") or ""
    if answer:
        flow.append(Paragraph("<b>What the AI assistant said:</b>", H3_STYLE))
        flow.append(Paragraph(escape(answer), BODY_STYLE))

    confidence = ai_answer.get("confidence")
    if confidence:
        flow.append(Spacer(1, 4))
        flow.append(Paragraph(f"<b>Confidence in what it could tell a customer:</b> {escape(str(confidence))}", BODY_STYLE))

    missing = ai_answer.get("missing") or []
    if missing:
        flow.append(Paragraph("What the AI couldn't find or was unsure about:", H3_STYLE))
        flow.append(ListFlowable(
            [ListItem(Paragraph(escape(m), ITEM_STYLE), spaceBefore=3) for m in missing],
            bulletType="bullet", leftIndent=14, bulletFontSize=9,
        ))
    return flow


def _page_detail_flowables(row: dict) -> list:
    """One page's full technical breakdown: header, on-page meta, schema
    found, contact/trust signal flags, and every issue on that page — not
    just the single top issue the free scan's page table shows."""
    header = f'{escape(str(row.get("url", "")))} — Score {row.get("score", "—")}/100 · Status {escape(str(row.get("status", "—")))}'
    flowables = [Paragraph(header, PAGE_HEADER_STYLE)]

    title = row.get("title") or "(missing title)"
    meta_description = row.get("meta_description") or "(missing meta description)"
    h1 = row.get("h1") or "(missing H1)"
    flowables.append(Paragraph(
        f"Title: {escape(title)}<br/>Meta description: {escape(meta_description)}<br/>H1: {escape(h1)}",
        PAGE_META_STYLE,
    ))

    schema_types = row.get("schema_types") or []
    schema_line = ", ".join(schema_types) if schema_types else "none found on this page"
    flowables.append(Paragraph(f"Schema types: {escape(schema_line)}", PAGE_META_STYLE))

    flags = [
        ("Phone", row.get("phone_ok")),
        ("Address", row.get("address_ok")),
        ("Hours", row.get("hours_ok")),
        ("Click-to-call", row.get("tel_link")),
        ("GBP/Maps link", row.get("gbp_link")),
        ("Mobile viewport", row.get("viewport_ok")),
    ]
    flags_line = " &nbsp;·&nbsp; ".join(f"{label}: {'Yes' if value else 'No'}" for label, value in flags)
    flowables.append(Paragraph(flags_line, PAGE_FLAG_STYLE))

    issues = row.get("issues") or []
    if issues:
        flowables.append(_bulleted(issues, ordered=False))
    else:
        flowables.append(Paragraph("No issues found on this page.", ITEM_STYLE))

    flowables.append(Spacer(1, 4))
    flowables.append(_hr())
    return flowables


def build_pdf_report(
    *,
    site_label: str,
    scanned_at: str,
    score: int,
    grade: str,
    band_label: str,
    message: str,
    benchmark_text: str,
    recommendations: list[str],
    gaps: list[str],
    robots_found: bool,
    sitemap_found: bool,
    llms_found: bool,
    schema_labels: list[str],
    pages_rows: list[dict],
    missing_schema_urls: list[str] | None = None,
    category_scores: list[dict] | None = None,
    snippet_url: str | None = None,
    snippet_phone: str | None = None,
    include_schema_snippet: bool = False,
    watermark: str | None = None,
    discovery: dict | None = None,
    ai_answer: dict | None = None,
) -> bytes:
    """Render the full LocalAgentReady report as PDF bytes.

    pages_rows: list of per-page dicts — url, score, status, title,
        meta_description, h1, schema_types (list[str]), issues (list[str]),
        phone_ok, address_ok, hours_ok, tel_link, gbp_link, viewport_ok.
    category_scores: list of {title, score, status_label, description} — the
        technical category breakdown, shown in the PDF (too jargon-heavy for
        the on-screen report the business owner reads).
    missing_schema_urls: URLs of scanned pages with zero structured data at
        all — the exact pages behind the "N of M pages..." recommendation.
    snippet_url / snippet_phone: real values to personalize the ready-to-
        paste schema snippet with, when include_schema_snippet is True.
    watermark: short label (e.g. "EXAMPLE — SAMPLE DATA") shown under the
        title; used for the fabricated example report so it can never be
        mistaken for a real scan.
    discovery: optional live "does AI recommend you?" result —
        {query, appears (bool), position (int|None), verdict (str),
        recommended: [{name, is_you}]}. Rendered near the top when present.
    ai_answer: optional AI-assistant simulation —
        {question, answer, confidence, missing: [str]}. Rendered when present.
    """
    missing_schema_urls = missing_schema_urls or []
    category_scores = category_scores or []

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title="LocalAgentReady Full Report",
    )

    story = [
        Paragraph("LOCALAGENTREADY", BRAND_STYLE),
        Paragraph("Full AI Readiness Report", TITLE_STYLE),
    ]
    if watermark:
        story.append(Paragraph(escape(watermark.upper()), WATERMARK_STYLE))
    story.append(Spacer(1, 6))
    story.append(Paragraph(escape(site_label), SITE_LABEL_STYLE))
    story.append(Paragraph(f"Scanned {escape(scanned_at)}", META_STYLE))
    story.append(Spacer(1, 14))

    score_table = Table(
        [[Paragraph(f"{score}/100", SCORE_STYLE), Paragraph(f"Grade {escape(str(grade))}", SCORE_STYLE)]],
        colWidths=[3.2 * inch, 3.2 * inch],
    )
    score_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM")]))
    story.append(score_table)
    story.append(Paragraph(escape(band_label), BAND_STYLE))
    story.append(Spacer(1, 10))
    story.append(Paragraph(escape(message), BODY_STYLE))
    story.append(Spacer(1, 4))
    story.append(Paragraph(escape(benchmark_text), META_STYLE))

    # The live AI results lead the report — they're the paid tier's headline,
    # answering "is AI recommending me?" directly rather than by proxy.
    if discovery:
        story.extend(_ai_discovery_flowables(discovery))
    if ai_answer:
        story.extend(_ai_answer_flowables(ai_answer))

    story.append(Paragraph("Top things to fix", H2_STYLE))
    story.append(_hr())
    story.append(_bulleted_recommendations(recommendations))

    if category_scores:
        story.append(Paragraph("Category breakdown", H2_STYLE))
        story.append(_hr())
        cat_rows = [[
            Paragraph("Category", TABLE_HEAD_STYLE),
            Paragraph("Score", TABLE_HEAD_STYLE),
            Paragraph("Status", TABLE_HEAD_STYLE),
            Paragraph("Detail", TABLE_HEAD_STYLE),
        ]]
        for c in category_scores:
            cat_rows.append([
                Paragraph(f"<b>{escape(str(c.get('title', '')))}</b>", TABLE_CELL_STYLE),
                Paragraph(f"{c.get('score', '')}", TABLE_CELL_STYLE),
                Paragraph(escape(str(c.get('status_label', ''))), TABLE_CELL_STYLE),
                Paragraph(escape(str(c.get('description', ''))), TABLE_CELL_STYLE),
            ])
        cat_table = Table(cat_rows, colWidths=[1.35 * inch, 0.5 * inch, 0.75 * inch, 3.6 * inch], repeatRows=1)
        cat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("GRID", (0, 0), (-1, -1), 0.5, LINE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
        ]))
        story.append(cat_table)

    if missing_schema_urls:
        story.append(Paragraph("Pages with no structured data at all", H2_STYLE))
        story.append(_hr())
        story.append(Paragraph(
            "These pages carry none of your business's schema — fixing the pages below first closes most of "
            "the gap behind the recommendation above:",
            BODY_STYLE,
        ))
        story.append(Spacer(1, 4))
        story.append(ListFlowable(
            [ListItem(Paragraph(escape(url), MISSING_SCHEMA_ITEM_STYLE), spaceBefore=3) for url in missing_schema_urls],
            bulletType="bullet",
            leftIndent=14,
            bulletFontSize=9,
        ))

    if include_schema_snippet:
        story.append(Paragraph("Ready-to-paste schema example", H3_STYLE))
        personalized = bool(snippet_url or snippet_phone)
        note = (
            "Personalized with the URL and phone number this scan found — replace the remaining "
            "placeholders (name, address, hours, profiles) with your real details:"
            if personalized else
            "Add this inside a script tag in your homepage's &lt;head&gt;, then replace the "
            "placeholder values with your real business details:"
        )
        story.append(Paragraph(note, BODY_STYLE))
        story.append(Preformatted(_schema_snippet(snippet_url or "https://yourbusiness.com", snippet_phone), CODE_STYLE))

    story.append(Paragraph("What may not be clear", H2_STYLE))
    story.append(_hr())
    story.append(_bulleted(gaps, ordered=False))

    story.append(Paragraph("Website access check", H2_STYLE))
    story.append(_hr())
    story.append(_kv_table([
        ("Search access file (robots.txt)", "Found" if robots_found else "Not found"),
        ("Website map (sitemap.xml)", "Found" if sitemap_found else "Not found"),
        ("AI assistant summary (llms.txt)", "Found" if llms_found else "Not found (optional, emerging standard)"),
    ]))

    if schema_labels:
        story.append(Paragraph("Business details found", H2_STYLE))
        story.append(_hr())
        story.append(Paragraph(escape(", ".join(schema_labels)), BODY_STYLE))

    story.append(Paragraph("Page-by-page technical audit", H2_STYLE))
    story.append(_hr())
    story.append(Paragraph(
        f"Full detail for all {len(pages_rows)} pages scanned — every issue found, not just the top one, "
        "plus the on-page signals AI tools and crawlers actually read.",
        BODY_STYLE,
    ))
    for row in pages_rows:
        for flowable in _page_detail_flowables(row):
            story.append(flowable)

    doc.build(story)
    return buffer.getvalue()
