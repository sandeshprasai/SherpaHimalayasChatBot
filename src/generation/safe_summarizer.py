#!/usr/bin/env python3
"""
Lightweight summarizer used when the LLM backend is unavailable.
It composes a concise, human-friendly answer from the top-k retrieved
source nodes so the chatbot can still answer free-form queries.
"""
from __future__ import annotations
import textwrap
from typing import List

from config.settings import settings


def _first_paragraph(text: str, max_len: int = 400) -> str:
    if not text:
        return ""
    parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    first = parts[0] if parts else text.strip()
    return (first[:max_len] + '...') if len(first) > max_len else first


def safe_summarize_nodes(nodes: List[object], query: str, max_chars: int = 1800) -> str:
    """Return a stitched summary from retrieved nodes.

    nodes: list of objects with attributes .title (optional), .text (string), .score (optional), .source_id (optional)
    query: original user query (for context)
    """
    if not nodes:
        return (
            f"Namaste. I couldn't find information on that topic in our records — "
            f"please contact {settings.company_email} for assistance."
        )

    parts = []
    parts.append(f"Summary (based on {len(nodes)} source(s)) for: {query}\n")

    for i, node in enumerate(nodes[:5], start=1):
        title = getattr(node, 'title', None) or getattr(node, 'source_id', None) or f"Document {i}"
        score = getattr(node, 'score', None)
        heading = f"{i}. {title}"
        if score is not None:
            heading += f" (relevance: {score:.2f})"
        parts.append(heading)

        text = getattr(node, 'text', '') or ''
        excerpt = _first_paragraph(text, max_len=400)
        parts.append(textwrap.fill(excerpt, width=100))

        # Extract a few logistics/booking lines if present
        booking_kws = ("book", "booking", "how to book", "contact", "includes", "inclusions", "price", "cost", "permit", "transport", "pickup", "drop", "arrival")
        booking_lines = []
        for line in text.splitlines():
            ll = line.lower()
            if any(k in ll for k in booking_kws):
                l = line.strip()
                if l and l not in booking_lines:
                    booking_lines.append(l)
        if booking_lines:
            parts.append("  Extracted logistics:")
            for bl in booking_lines[:4]:
                parts.append(f"   - {bl}")

        parts.append("")

    # add source listing
    parts.append("Sources:")
    for i, node in enumerate(nodes[:5], start=1):
        title = getattr(node, 'title', None) or getattr(node, 'source_id', None) or f"Document {i}"
        src = getattr(node, 'source_id', None) or "N/A"
        parts.append(f"- {title} ({src})")

    out = "\n".join(parts)
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "..."
    return out
