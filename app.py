import csv
import io
import json
import re
from dataclasses import asdict
from datetime import datetime

import pandas as pd
import streamlit as st

from ai_readiness_scanner import crawl


st.set_page_config(
    page_title="AIReady Scanner",
    page_icon="✅",
    layout="wide",
)


def normalize_url_for_display(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    return url


def report_to_json_bytes(report) -> bytes:
    return json.dumps(asdict(report), indent=2, ensure_ascii=False).encode("utf-8")


def pages_to_dataframe(report) -> pd.DataFrame:
    rows = []
    for p in report.pages:
        rows.append({
            "URL": p.url,
            "Status": p.status_code,
            "Score": p.page_score,
            "Title": p.title,
            "Meta description": p.meta_description,
            "Schema types": ", ".join(p.schema_types),
            "Phone": p.phone_found or p.schema_has_phone,
            "Address": p.address_like_found or p.schema_has_address,
            "Hours": p.hours_like_found or p.schema_has_hours,
            "Load ms": p.load_ms,
            "Size KB": p.size_kb,
            "Top issues": " | ".join(p.issues[:5]),
        })
    return pd.DataFrame(rows)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def grade_message(score: int) -> str:
    if score >= 85:
        return "Strong AI readiness. The site is relatively easy for AI systems to understand."
    if score >= 70:
        return "Good foundation. A few improvements could make the business clearer to AI systems."
    if score >= 55:
        return "Partially ready. AI can understand some basics, but important trust and business signals are missing."
    if score >= 40:
        return "Weak readiness. The site likely needs clearer structure, schema, and business details."
    return "Not AI-ready yet. The site may be hard for AI systems to understand or recommend confidently."


def collect_site_gaps(report) -> list[str]:
    gaps = []

    schema_counts = report.schema_type_counts or {}
    has_business_schema = any(
        t in schema_counts
        for t in [
            "LocalBusiness",
            "Organization",
            "ProfessionalService",
            "HomeAndConstructionBusiness",
            "Plumber",
            "Electrician",
            "HVACBusiness",
            "RoofingContractor",
            "GeneralContractor",
            "Dentist",
            "LegalService",
        ]
    )

    homepage = report.pages[0] if report.pages else None

    if not has_business_schema:
        gaps.append("AI may not clearly identify the business type because strong LocalBusiness or Organization schema was not found.")

    if homepage:
        if not (homepage.schema_has_phone or homepage.phone_found):
            gaps.append("AI may not see a clear phone number.")
        if not (homepage.schema_has_address or homepage.address_like_found):
            gaps.append("AI may not understand the business address or service area.")
        if not (homepage.schema_has_hours or homepage.hours_like_found):
            gaps.append("AI may not know when the business is open.")
        if len(homepage.service_words_found) < 3:
            gaps.append("AI may not clearly understand the services offered or booking intent.")
        if not homepage.has_faq_schema:
            gaps.append("AI may not have direct answers to common customer questions because FAQ schema was not found.")

    if not gaps:
        gaps.append("The main business signals are present. The next step is improving depth, reviews, FAQs, and service-area content.")

    return gaps


def render_score_card(report):
    st.subheader("AI Readiness Result")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Readiness Score", f"{report.site_score}/100")
    with col2:
        st.metric("Grade", report.grade)
    with col3:
        st.metric("Pages scanned", report.pages_scanned)

    st.info(grade_message(report.site_score))


def render_recommendations(report):
    st.subheader("Top Fixes")
    for i, rec in enumerate(report.top_recommendations[:5], start=1):
        st.write(f"**{i}.** {rec}")


def render_ai_understanding(report):
    st.subheader("What AI May Not Understand")
    for gap in collect_site_gaps(report):
        st.write(f"- {gap}")


def render_schema_section(report):
    st.subheader("Structured Data Found")
    if report.schema_type_counts:
        st.json(report.schema_type_counts)
    else:
        st.warning("No Schema.org JSON-LD types were found.")


def render_lead_capture(report):
    st.subheader("Request Help Fixing This")
    st.caption("MVP lead capture: this stores the lead in a downloadable CSV in your browser session. For production, connect this to Tally, Airtable, Google Sheets, or a CRM.")

    with st.form("lead_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        business_type = st.text_input("Business type", placeholder="Example: plumber, dentist, remodeler")
        notes = st.text_area("Notes", placeholder="What do you want help with?")
        submitted = st.form_submit_button("Create Lead Record")

    if submitted:
        lead = {
            "created_at": datetime.utcnow().isoformat(),
            "name": name,
            "email": email,
            "business_type": business_type,
            "website": report.normalized_url,
            "score": report.site_score,
            "grade": report.grade,
            "notes": notes,
        }
        st.success("Lead record created. Download it below.")
        st.download_button(
            "Download lead CSV",
            data=pd.DataFrame([lead]).to_csv(index=False).encode("utf-8"),
            file_name="aiready_lead.csv",
            mime="text/csv",
        )


def main():
    st.title("AIReady Scanner")
    st.write("A simple MVP scanner for AI visibility readiness.")

    with st.expander("What this checks", expanded=False):
        st.write(
            """
            This MVP checks crawlability, metadata, Schema.org JSON-LD, business/contact signals,
            service language, social/profile links, Open Graph metadata, image alt text, and basic page-size/load indicators.
            """
        )

    with st.sidebar:
        st.header("Scan Settings")
        max_pages = st.slider("Max pages to scan", min_value=1, max_value=30, value=10)
        st.caption("For the MVP, keep this low so scans finish quickly.")

    url = st.text_input("Website URL", placeholder="https://example.com")

    scan = st.button("Scan My Site", type="primary")

    if scan:
        if not url.strip():
            st.error("Please enter a website URL.")
            return

        scan_url = normalize_url_for_display(url)

        with st.spinner("Scanning website..."):
            try:
                report = crawl(scan_url, max_pages=max_pages)
            except Exception as exc:
                st.error(f"Scan failed: {exc}")
                return

        st.session_state["report"] = report

    report = st.session_state.get("report")

    if report:
        st.divider()
        render_score_card(report)

        left, right = st.columns([1, 1])
        with left:
            render_recommendations(report)
        with right:
            render_ai_understanding(report)

        st.divider()

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.subheader("Crawl Basics")
            st.write(f"**robots.txt:** {report.robots_status} — {report.robots_url}")
            st.write(f"**sitemap.xml:** {report.sitemap_status} — {report.sitemap_url}")
        with col_b:
            render_schema_section(report)

        st.divider()

        st.subheader("Page Details")
        df = pages_to_dataframe(report)
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download JSON Report",
            data=report_to_json_bytes(report),
            file_name="aiready_report.json",
            mime="application/json",
        )

        st.download_button(
            "Download CSV Page Report",
            data=dataframe_to_csv_bytes(df),
            file_name="aiready_pages.csv",
            mime="text/csv",
        )

        st.divider()
        render_lead_capture(report)

    else:
        st.divider()
        st.subheader("Recommended MVP positioning")
        st.write("**AIReady** helps local businesses see whether their websites are ready to be found, understood, and recommended by AI.")
        st.write("Start with local service businesses: plumbers, electricians, HVAC, remodelers, roofers, dentists, and cleaners.")


if __name__ == "__main__":
    main()
