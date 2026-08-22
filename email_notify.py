#!/usr/bin/env python3
"""Email notifications for LocalAgentReady, via Resend.

The one email that matters right now: when someone requests the full report,
YOU (the owner) get an email telling you which website they scanned, their
score, and how to reach them — so a manual-delivery flow (you email the PDF,
they pay via the Stripe link) has everything it needs to close the loop.

FAILS SOFT like datastore.py: no key, no library, or an API error just means
no email got sent and the caller is told so — the lead is still logged to the
sheet and shown on screen, so nothing is lost.

Setup (Streamlit secrets):
  RESEND_API_KEY = "re_..."           # required to send anything
  LEAD_NOTIFY_TO = "you@example.com"  # where lead alerts go (default below)
  RESEND_FROM    = "LocalAgentReady <onboarding@resend.dev>"  # verified sender
"""

from __future__ import annotations

from html import escape

import streamlit as st

# Resend's shared sandbox sender works with zero domain setup, but can ONLY
# deliver to the email tied to your Resend account. Fine for owner alerts to
# yourself; set RESEND_FROM to a verified domain before emailing customers.
_DEFAULT_FROM = "LocalAgentReady <onboarding@resend.dev>"
_DEFAULT_TO = "rex@liftlogic.com"


def _secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default)) or default
    except Exception:
        return default


def is_available() -> bool:
    """True only if the library is importable and an API key is configured."""
    try:
        import resend  # noqa: F401
    except Exception:
        return False
    return bool(_secret("RESEND_API_KEY"))


def notify_recipient() -> str:
    return _secret("LEAD_NOTIFY_TO", _DEFAULT_TO)


def send_lead_notification(
    *,
    website: str,
    score,
    grade: str,
    company: str = "",
    contact_name: str = "",
    phone: str = "",
    email: str = "",
    payment_link: str = "",
) -> tuple[bool, str]:
    """Email the owner that someone wants the full report. Returns
    (sent, message) — never raises, so a failed send can't break the form."""
    if not is_available():
        return False, "Email is not configured on the server."

    try:
        import resend

        resend.api_key = _secret("RESEND_API_KEY")

        rows = [
            ("Website scanned", website),
            ("AI-readiness score", f"{score}/100 (grade {grade})"),
            ("Company", company or "—"),
            ("Contact name", contact_name or "—"),
            ("Phone", phone or "—"),
            ("Email", email or "—"),
        ]
        rows_html = "".join(
            f'<tr><td style="padding:6px 12px;color:#5f6368;white-space:nowrap;">{escape(label)}</td>'
            f'<td style="padding:6px 12px;color:#0b0b0f;font-weight:600;">{escape(str(value))}</td></tr>'
            for label, value in rows
        )
        pay_html = (
            f'<p style="margin-top:16px;">Payment link to send them: '
            f'<a href="{escape(payment_link)}">{escape(payment_link)}</a></p>'
            if payment_link
            else ""
        )
        html = (
            '<div style="font-family:Arial,Helvetica,sans-serif;max-width:560px;">'
            '<h2 style="color:#0b0b0f;">New full-report request</h2>'
            '<table style="border-collapse:collapse;font-size:14px;">'
            f"{rows_html}</table>{pay_html}"
            '<p style="color:#9a9894;font-size:12px;margin-top:20px;">'
            "Sent automatically by LocalAgentReady.</p></div>"
        )

        payload: dict = {
            "from": _secret("RESEND_FROM", _DEFAULT_FROM),
            "to": [notify_recipient()],
            "subject": f"Report request — {website} ({score}/100)",
            "html": html,
        }
        # So you can reply straight to the customer from the alert.
        if email:
            payload["reply_to"] = email

        resend.Emails.send(payload)
        return True, "Notification sent."
    except Exception as exc:
        return False, f"Email send failed: {exc}"
