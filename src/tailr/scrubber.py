"""PII scrubber — removes personal identifiable information from CV text.

Strips names, emails, phone numbers, LinkedIn/GitHub URLs, generic URLs,
and street addresses before the CV is sent to an LLM.  Uses only regex
and heuristics — no external NLP dependencies.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Patterns — ordered from most-specific to least-specific so that e.g.
# LinkedIn URLs are caught before the generic URL pattern.
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
)

_PHONE_RE = re.compile(
    r"(?<!\d)"  # not preceded by a digit
    r"(?:\+\d{1,3}[\s\-.]?)?"  # optional country code
    r"(?:\(?\d{1,4}\)?[\s\-.]?)?"  # optional area code
    r"\d[\d\s\-.]{6,14}\d"  # core number (7–15 digits with separators)
    r"(?!\d)",  # not followed by a digit
)

_LINKEDIN_RE = re.compile(
    r"https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+/?",
    re.IGNORECASE,
)

_GITHUB_RE = re.compile(
    r"https?://(?:www\.)?github\.com/[A-Za-z0-9\-_%]+/?",
    re.IGNORECASE,
)

_URL_RE = re.compile(
    r"https?://[^\s<>\"')\]]+",
)

# Matches lines like "123 Main St", "456 Elm Avenue, Springfield, IL 62704"
_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+"
    r"[A-Za-z0-9\s.]+?"
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl)"
    r"\b"
    r"[^,\n]*"  # rest of line (city, state, zip …)
    r"(?:,\s*[A-Za-z\s]+(?:\d{4,5}(?:-\d{4})?)?)?",
    re.IGNORECASE,
)

# Common header words that should NOT be treated as a person's name.
_HEADER_WORDS = frozenset(
    {
        "curriculum vitae",
        "cv",
        "resume",
        "résumé",
        "profile",
        "summary",
        "about me",
        "personal statement",
    }
)


def _scrub_name(text: str) -> str:
    """Replace the candidate's name (first non-trivial line) with [NAME].

    Skips blank lines and common CV header words so we don't accidentally
    replace "Curriculum Vitae" instead of the actual name.
    """
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        stripped = line.strip().strip("#").strip()
        if not stripped:
            continue
        if stripped.lower() in _HEADER_WORDS:
            continue
        # Skip lines that are too long to be a name.
        if len(stripped) > 60:
            continue
        # Skip lines containing obvious non-name characters.
        if any(ch in stripped for ch in ("@", "://", ":", "+", "(")):
            continue
        # A name should be 2–5 words, each mostly alphabetic.
        words = stripped.split()
        if not (2 <= len(words) <= 5):
            continue
        if not all(re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ\-'.]+", w) for w in words):
            continue
        # Replace this name throughout the document.
        name_pattern = re.compile(re.escape(stripped))
        text = name_pattern.sub("[NAME]", text)
        break
    return text


def scrub(text: str) -> str:
    """Return *text* with PII replaced by placeholder tokens.

    Replacement order matters — more-specific URL patterns run before the
    generic catch-all so LinkedIn and GitHub links get their own tokens.
    """
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _LINKEDIN_RE.sub("[LINKEDIN]", text)
    text = _GITHUB_RE.sub("[GITHUB]", text)
    text = _URL_RE.sub("[URL]", text)
    text = _ADDRESS_RE.sub("[ADDRESS]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _scrub_name(text)
    return text
