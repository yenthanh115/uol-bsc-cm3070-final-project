#!/usr/bin/env python
"""Build the ticker-extraction stopword lexicon from a reproducible recipe.

WHY THIS SCRIPT EXISTS
----------------------
Ticker extraction (``surge_pipeline.loader``) casts a broad net: any standalone
2-5 character uppercase token can look like a stock ticker (``PLTR``, ``AMC``),
but so can ordinary uppercase words (``THE``, ``CEO``, ``YOLO``). To suppress
those false positives the loader filters candidates against a stopword lexicon.

this script *derives* the lexicon from two clearly-separated sources:

  1. ENGLISH BASE (mechanical, public provenance)
     The bulk of common English words comes straight from the NLTK ``stopwords``
     corpus. We upper-case it and keep only tokens the ticker regex could match
     (1-5 characters, A-Z only). This part is fully reproducible from a public
     corpus -- no manual curation, no mystery.

  2. DOMAIN SUPPLEMENT (curated, error-analysis provenance)
     A finite, categorised set of finance / Reddit / market terms that are NOT
     general-English stopwords but still routinely masquerade as tickers in
     r/wallstreetbets and r/pennystocks posts (``DD``, ``YOLO``, ``NASDAQ``,
     ``OTCQB``, ``CEO`` ...), plus the handful of everyday words the NLTK list
     omits (``HUGE``, ``TECH``, ``STOCK`` ...). These were identified by an
     iterative error-analysis loop on early extraction runs: run extraction,
     inspect the most frequent uppercase false positives, assign each to a
     category below, and repeat until the false-positive rate stabilised.

The final lexicon is ``ENGLISH_BASE | DOMAIN_SUPPLEMENT``, written to a
self-documenting data file that the loader reads at import time. Re-run this
script whenever the recipe changes; the data file is the build artefact and the
loader never hard-codes the words again.

USAGE
-----
    python src/build_stopwords.py            # writes the default data file
    python src/build_stopwords.py --check    # verify file is up to date (CI)

The default output path is ``src/surge_pipeline/data/ticker_stopwords.txt``,
inside the package so the lexicon ships with the installed wheel.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _PROJECT_ROOT / "src" / "surge_pipeline" / "data" / "ticker_stopwords.txt"

# Only tokens the ticker regexes could ever produce are worth filtering.
_TICKER_SHAPED = re.compile(r"^[A-Z]{1,5}$")


# --------------------------------------------------------------------------- #
# Source 1: English base (mechanical, from the NLTK stopwords corpus)
# --------------------------------------------------------------------------- #
def build_english_base() -> set[str]:
    """Return upper-cased NLTK English stopwords filtered to ticker-shaped tokens.

    Downloads the corpus on first use so the build is reproducible on a fresh
    machine without a manual setup step.
    """
    try:
        import nltk
        from nltk.corpus import stopwords
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "nltk is required to build the stopword lexicon. "
            "Install it with `pip install nltk` (declared in pyproject.toml)."
        ) from exc

    # Idempotent; no-op once the corpus is present.
    nltk.download("stopwords", quiet=True)

    words = {w.upper() for w in stopwords.words("english")}
    return {w for w in words if _TICKER_SHAPED.match(w)}


# --------------------------------------------------------------------------- #
# Source 2: Domain supplement (curated, categorised, error-analysis provenance)
#
# These are terms that are NOT general-English stopwords but still show up as
# uppercase false-positive "tickers" in financial-Reddit text, plus the small
# set of everyday words the NLTK corpus happens to omit. Grouped by category so
# the list stays auditable and easy to extend.
# --------------------------------------------------------------------------- #
DOMAIN_SUPPLEMENT: dict[str, set[str]] = {
    # Everyday English words the NLTK stopword corpus does not include but which
    # are common in posts and are not tickers.
    "common_english_extra": {
        "ALSO", "BACK", "BEST", "BIG", "BUY", "COME", "DAY", "DONE", "DONT",
        "EVEN", "FEEL", "FIND", "GET", "GIVE", "GO", "GOOD", "GOT", "HELP",
        "HIGH", "HOT", "HUGE", "INFO", "KEEP", "KNOW", "LEFT", "LESS", "LETS",
        "LIFE", "LIKE", "LINE", "LIST", "LOOK", "LOTS", "LOW", "MADE", "MAKE",
        "MANY", "MAY", "MUCH", "MUST", "NEED", "NEW", "OH", "OLD", "ONE",
        "OPEN", "PART", "PAST", "PLAN", "PUT", "REAL", "RED", "REST", "RUN",
        "SAID", "SAW", "SET", "SHOW", "SIDE", "SURE", "TAKE", "TECH", "TELL",
        "TEN", "TERM", "TIME", "TOP", "TWO", "TYPE", "USED", "WAIT", "WANT",
        "WELL", "WENT", "WORK",
    },
    # Reddit / trading-forum slang and post verbs that are not tickers.
    "reddit_slang": {
        "APE", "CALL", "DD", "DM", "DR", "DROP", "DUMP", "EDIT", "FALL",
        "FOMO", "FREE", "FUD", "GAIN", "HOLD", "ICYMI", "IMHO", "IMO", "LAST",
        "LINK", "LMAO", "LOL", "LONG", "LOSS", "MOON", "MOVE", "NEXT", "NSFW",
        "OMG", "OP", "PENNY", "PICK", "PLAY", "POST", "PRICE", "PUMP", "PUTS",
        "RISE", "ROFL", "SELL", "SHARE", "SHARES", "SHORT", "SMH", "STOCK",
        "TBH", "TL", "TLDR", "TRADE", "UPDATE", "WEEK", "WSB", "WTF", "YEAR",
        "YOLO",
    },
    # Corporate / financial abbreviations and entity suffixes.
    "finance_abbreviations": {
        "ATH", "ATL", "AVG", "CEO", "CFO", "CO", "CORP", "CPI", "EMA", "EOD",
        "EOM", "EOW", "EPS", "ETF", "FDA", "GDP", "INC", "IPO", "ITM", "LLC",
        "LTD", "MACD", "MAX", "MIN", "NYSE", "OTC", "OTM", "ROI", "RSI", "SEC",
        "SMA", "VS",
    },
    # Exchange names and market identifiers (never valid tickers in context).
    "market_venues": {
        "AMEX", "CSE", "NASDAQ", "OTCQB", "OTCQX", "SP", "TSX", "TSXV",
    },
    # Currencies and asset-class labels.
    "currencies_and_assets": {
        "BTC", "CAD", "CRYPTO", "EUR", "GBP", "USD",
    },
    # Time zones, weekdays, months and units.
    "datetime_and_units": {
        "APR", "AUG", "DEC", "EST", "FEB", "FRI", "FT", "JAN", "JUL", "JUN",
        "LB", "MAR", "MON", "NOV", "OCT", "OZ", "PM", "PST", "SAT", "SEP",
        "SUN", "THU", "TUE", "UTC", "WED",
    },
    # Country / region codes common in EU/US market chatter.
    "geography": {
        "CA", "EU", "FL", "NY", "PT", "TX", "UK", "US", "USA",
    },
    # Technology acronyms frequently mis-read as tickers.
    "technology_buzzwords": {
        "AI", "AR", "CBD", "COVID", "EV", "NFT", "VR",
    },
    # Remaining high-frequency false positives specific to penny-stock posts.
    "reddit_false_positives": {
        "GLOBE", "PINK", "PR", "XXXX", "ZERO",
    },
}


def build_supplement() -> set[str]:
    """Flatten the categorised domain supplement into a single set."""
    out: set[str] = set()
    for terms in DOMAIN_SUPPLEMENT.values():
        out |= terms
    return out


# --------------------------------------------------------------------------- #
# File rendering
# --------------------------------------------------------------------------- #
def render_file() -> str:
    """Render the full, self-documenting stopword data file as text."""
    english_base = build_english_base()

    header = [
        "# =====================================================================",
        "# ticker_stopwords.txt  --  GENERATED FILE, DO NOT EDIT BY HAND",
        "# =====================================================================",
        "#",
        "# Regenerate with:  python src/build_stopwords.py",
        f"# Generated (UTC):  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "#",
        "# WHAT THIS IS",
        "#   The stopword lexicon used by ticker extraction to reject uppercase",
        "#   tokens that look like tickers but are not (THE, CEO, YOLO, NASDAQ ...).",
        "#",
        "# HOW IT WAS BUILT (two sources, see src/build_stopwords.py)",
        "#   1. ENGLISH BASE  -- NLTK 'stopwords' corpus (English), upper-cased and",
        "#      filtered to ticker-shaped tokens (1-5 chars, A-Z). Public, mechanical",
        "#      provenance; no manual curation.",
        "#   2. DOMAIN SUPPLEMENT -- a curated, categorised set of finance / Reddit /",
        "#      market terms (and a few everyday words NLTK omits) identified by",
        "#      iterative error analysis on early extraction runs.",
        "#",
        "#   Final lexicon = ENGLISH BASE union DOMAIN SUPPLEMENT.",
        "#",
        "# FORMAT",
        "#   One term per line, upper-case. Blank lines and lines beginning with '#'",
        "#   are ignored by the loader. Section headers below are comments only.",
        "# =====================================================================",
        "",
    ]

    lines: list[str] = list(header)

    lines.append("# ---------------------------------------------------------------------")
    lines.append(f"# SECTION 1 -- English base (NLTK stopwords corpus, {len(english_base)} terms)")
    lines.append("# ---------------------------------------------------------------------")
    for term in sorted(english_base):
        lines.append(term)
    lines.append("")

    lines.append("# ---------------------------------------------------------------------")
    lines.append("# SECTION 2 -- Domain supplement (curated, by category)")
    lines.append("# ---------------------------------------------------------------------")
    # Only emit supplement terms not already provided by the English base, so
    # the file lists each term exactly once and the categories stay meaningful.
    for category, terms in DOMAIN_SUPPLEMENT.items():
        unique = sorted(terms - english_base)
        if not unique:
            continue
        lines.append(f"# [{category}] ({len(unique)} terms)")
        for term in unique:
            lines.append(term)
        lines.append("")

    total = len(english_base | build_supplement())
    lines.append(f"# Total unique terms in lexicon: {total}")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the ticker stopword lexicon.")
    parser.add_argument(
        "--output", "-o", type=Path, default=_DEFAULT_OUTPUT,
        help=f"Output path (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Do not write; exit non-zero if the existing file is out of date.",
    )
    args = parser.parse_args(argv)

    content = render_file()

    if args.check:
        if not args.output.exists():
            print(f"[build_stopwords] MISSING: {args.output}", file=sys.stderr)
            return 1
        current = args.output.read_text(encoding="utf-8")
        # Compare ignoring the generation-timestamp line, which always differs.
        def _strip_ts(text: str) -> list[str]:
            return [ln for ln in text.splitlines() if not ln.startswith("# Generated (UTC):")]
        if _strip_ts(current) != _strip_ts(content):
            print("[build_stopwords] OUT OF DATE: regenerate with python src/build_stopwords.py",
                  file=sys.stderr)
            return 1
        print("[build_stopwords] up to date.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    total = len(build_english_base() | build_supplement())
    print(f"[build_stopwords] wrote {total} terms to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
