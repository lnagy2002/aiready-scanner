#!/usr/bin/env python3
"""Google Sheets persistence for LocalAgentReady.

Two jobs, one spreadsheet:
  1. A rolling log of every real scan's score, so the "typical local-business
     score" benchmark can ADAPT to what we actually see instead of staying a
     hardcoded guess (see get_typical_score).
  2. A durable lead log — every "I want the full report" submission — so leads
     survive a Streamlit Cloud restart (its filesystem is ephemeral and wipes
     local files on every redeploy).

Everything here FAILS SOFT: if the dependency isn't installed, the secrets
aren't configured, or the Sheets API errors, the calling code gets a safe
fallback (False / None / the seed benchmark) and the app keeps working. None
of this is on the critical path of a scan.

Setup (Streamlit secrets):
  - A `[gcp_service_account]` table holding a Google service-account key.
  - `SHEET_ID = "..."` — the target spreadsheet's id (the long token in its URL).
  Share the spreadsheet with the service account's client_email as an Editor.
"""

from __future__ import annotations

import datetime as _dt

import streamlit as st

# The seed the benchmark starts from before enough real scans exist, and how
# heavily it's weighted. PRIOR_WEIGHT acts like "this many pretend scans at the
# seed value" — it keeps the number stable when only a handful of real scans
# have been logged, then lets the true average take over as volume grows.
BENCHMARK_SEED = 58
BENCHMARK_PRIOR_WEIGHT = 15

SCANS_WORKSHEET = "scans"
LEADS_WORKSHEET = "leads"

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def is_available() -> bool:
    """True only if both the library and the required secrets are present.
    Never raises — a missing secrets file must not crash the app."""
    try:
        import gspread  # noqa: F401
        from google.oauth2.service_account import Credentials  # noqa: F401
    except Exception:
        return False
    try:
        return bool(st.secrets.get("SHEET_ID") and st.secrets.get("gcp_service_account"))
    except Exception:
        return False


def availability_detail() -> dict:
    """Per-condition breakdown behind is_available(), so a failing setup can
    be diagnosed precisely (library missing vs. which secret is absent vs. a
    secrets-file parse error) instead of one opaque False."""
    detail: dict = {"library": False, "sheet_id": False, "service_account": False}
    try:
        import gspread  # noqa: F401
        from google.oauth2.service_account import Credentials  # noqa: F401

        detail["library"] = True
    except Exception as exc:
        detail["library_error"] = str(exc)
    try:
        detail["sheet_id"] = bool(st.secrets.get("SHEET_ID"))
        detail["service_account"] = bool(st.secrets.get("gcp_service_account"))
    except Exception as exc:
        detail["secrets_error"] = str(exc)
    return detail


@st.cache_resource(show_spinner=False)
def _spreadsheet():
    """Authorized handle to the spreadsheet, cached for the process. Returns
    None on any failure so callers can degrade gracefully."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=_SCOPES
        )
        client = gspread.authorize(creds)
        return client.open_by_key(st.secrets["SHEET_ID"])
    except Exception:
        return None


def _worksheet(title: str, headers: list[str]):
    """Fetch (or create) a worksheet by title, ensuring a header row exists."""
    sheet = _spreadsheet()
    if sheet is None:
        return None
    try:
        return sheet.worksheet(title)
    except Exception:
        try:
            ws = sheet.add_worksheet(title=title, rows=1000, cols=max(len(headers), 8))
            ws.append_row(headers)
            return ws
        except Exception:
            return None


def _now() -> str:
    # UTC, so rows from different environments/timezones sort consistently.
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log_scan(url: str, score: int, grade: str) -> bool:
    """Append one scan's result. Best-effort — returns False (never raises) if
    the store isn't available or the write fails."""
    if not is_available():
        return False
    ws = _worksheet(SCANS_WORKSHEET, ["timestamp", "url", "score", "grade"])
    if ws is None:
        return False
    try:
        ws.append_row([_now(), url, int(score), grade], value_input_option="RAW")
        _read_scores.clear()  # new data point — invalidate the cached average
        return True
    except Exception:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def _read_scores() -> list[int]:
    """Numeric scores from the scans worksheet, cached for 5 minutes so the
    benchmark doesn't cost a Sheets read on every rerun."""
    ws = _worksheet(SCANS_WORKSHEET, ["timestamp", "url", "score", "grade"])
    if ws is None:
        return []
    try:
        raw = ws.col_values(3)[1:]  # column C ("score"), skipping the header
    except Exception:
        return []
    scores: list[int] = []
    for value in raw:
        try:
            scores.append(int(float(value)))
        except (TypeError, ValueError):
            continue
    return scores


def get_typical_score(seed: int = BENCHMARK_SEED, prior_weight: int = BENCHMARK_PRIOR_WEIGHT) -> int:
    """The adaptive benchmark. Starts at `seed`, then blends toward the real
    average of logged scans as more accumulate. Falls back to `seed` whenever
    the store is unavailable, so the UI always has a number to show."""
    if not is_available():
        return seed
    scores = _read_scores()
    if not scores:
        return seed
    blended = (seed * prior_weight + sum(scores)) / (prior_weight + len(scores))
    return round(blended)


def log_lead(lead: dict) -> bool:
    """Append one report-purchase request. Best-effort; returns success."""
    if not is_available():
        return False
    headers = ["timestamp", "company", "contact_name", "phone", "email", "website", "score", "grade"]
    ws = _worksheet(LEADS_WORKSHEET, headers)
    if ws is None:
        return False
    try:
        ws.append_row(
            [
                _now(),
                lead.get("company", ""),
                lead.get("contact_name", ""),
                lead.get("phone", ""),
                lead.get("email", ""),
                lead.get("website", ""),
                lead.get("score", ""),
                lead.get("grade", ""),
            ],
            value_input_option="RAW",
        )
        return True
    except Exception:
        return False
