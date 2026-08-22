#!/usr/bin/env python3
"""Simulates what an AI assistant would actually say about a business when
a customer asks, using ONLY the text a real crawl found.

This is the product's actual promise ("is AI finding your business?")
tested directly against a real model, instead of only inferred from
schema/meta-tag heuristics — no other AI-readiness scanner does this,
since it requires an LLM call rather than a crawler.

Requires an Anthropic API key available to the `anthropic` SDK's default
credential resolution (an ANTHROPIC_API_KEY environment variable, or an
`ant auth login` profile) — see is_available().
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

import anthropic
from pydantic import BaseModel

# Kept as a plain module constant (not buried in the request builder) so
# it's easy to find and swap — e.g. to a cheaper model if this runs at
# meaningful volume. See the Anthropic API skill/docs for current model IDs.
MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are simulating how a helpful AI assistant (like ChatGPT, Claude, or Perplexity) "
    "would answer a potential customer who is deciding whether to contact this local "
    "business. Base your answer ONLY on the webpage content the user provides — do not "
    "use any outside knowledge, assumptions, or general reputation information about this "
    "business, its industry, or businesses like it. If the page does not clearly state "
    "something a customer would reasonably want to know, say plainly that it's unclear or "
    "not mentioned instead of guessing or filling in a plausible-sounding default."
)


class AIAssistantSimulation(BaseModel):
    answer: str
    confidence: Literal["High", "Medium", "Low"]
    missing_or_unclear: list[str]


def is_available() -> bool:
    """Whether this feature can actually run — i.e. whether the anthropic
    SDK has credentials to resolve at all. Doesn't guarantee the key is
    valid, just that one is configured, so callers can hide the feature
    entirely rather than surface a confusing failure."""
    import os

    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _site_label(url: str) -> str:
    netloc = urlparse(url).netloc.replace("www.", "")
    return netloc or url


def build_customer_question(url: str) -> str:
    """One consistent, generic question — grounded in the URL rather than a
    guessed business name (title-based name guessing is unreliable enough
    that a wrong guess would undercut the simulation's credibility)."""
    label = _site_label(url)
    return (
        f'I found "{label}" while searching for a local business — can you tell me what '
        "they offer, whether they seem trustworthy, where they're located, and how I would "
        "contact them?"
    )


def build_page_summary(*, title: str | None, meta_description: str | None, h1: str | None, schema_types: list[str]) -> str:
    return (
        f"Title: {title or '(missing)'}\n"
        f"Meta description: {meta_description or '(missing)'}\n"
        f"Main heading (H1): {h1 or '(missing)'}\n"
        f"Structured business data found on this page: {', '.join(schema_types) or 'none'}"
    )


def simulate_ai_answer(
    *,
    page_summary: str,
    visible_text_excerpt: str,
    customer_question: str,
) -> AIAssistantSimulation:
    """Runs the actual simulation. Raises anthropic's exception classes on
    failure (AuthenticationError, RateLimitError, APIStatusError,
    APIConnectionError, ...) — this module doesn't know about Streamlit, so
    it doesn't decide how those get surfaced; the caller does."""
    client = anthropic.Anthropic()

    user_prompt = f"""Business website content (homepage):
{page_summary}

Full visible page text (may be truncated):
\"\"\"
{visible_text_excerpt}
\"\"\"

A potential customer asks an AI assistant: "{customer_question}"

Answer as the AI assistant would, using ONLY the page content above."""

    response = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        output_format=AIAssistantSimulation,
    )
    return response.parsed_output
