"""
separator.py
------------
Content-based automatic separator for scraped Markdown files.

Input  : data/processed/raw_md/*.md
Output : data/separated/
            destinations/treks/
            destinations/expeditions/
            destinations/peaks/
            destinations/tours/
            destinations/regions/
            blogs/
            information/
            company/
            team/
            testimonials/
            junk/

Classification logic (pure heuristic -- zero tokens, zero API calls):
  Priority 1  -> JUNK        (size, known slugs, noise patterns)
  Priority 2  -> TEAM        (guide/staff profile pages)
  Priority 3  -> TESTIMONIALS
  Priority 4  -> COMPANY
  Priority 5  -> BLOGS       (exact name / URL path)
  Priority 6  -> INFORMATION (exact name / URL path / filename keyword)
  Priority 5b -> BLOGS       (content signals, after info exact overrides)
  Priority 6b -> INFORMATION (content signals fallback)
  Priority 7  -> DESTINATIONS/EXPEDITIONS
  Priority 8  -> DESTINATIONS/PEAKS
  Priority 9  -> DESTINATIONS/TREKS
  Priority 10 -> DESTINATIONS/TOURS
  Priority 11 -> DESTINATIONS/REGIONS
  Fallback    -> junk

Usage
-----
    python separator.py              # copy mode (default)
    python separator.py --move       # move files instead of copying
    python separator.py --dry-run    # print only, no file I/O
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE      = Path(__file__).parent
SOURCE_DIR = (_HERE / "../../data/processed/raw_md").resolve()
TARGET_DIR = (_HERE / "../../data/separated").resolve()

# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------
CATEGORIES = [
    "destinations/treks",
    "destinations/expeditions",
    "destinations/peaks",
    "destinations/tours",
    "destinations/regions",
    "blogs",
    "information",
    "company",
    "team",
    "testimonials",
    "junk",
]

for _cat in CATEGORIES:
    (TARGET_DIR / _cat).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> tuple[str, str]:
    """Return (raw_text, lowercase_text)."""
    try:
        raw = path.read_text(encoding="utf-8")
        return raw, raw.lower()
    except Exception:
        return "", ""


def _extract_url(raw: str) -> str:
    m = re.search(r"#\s*URL:\s*(https?://\S+)", raw)
    return m.group(1).lower() if m else ""


def _url_path(url: str) -> str:
    m = re.match(r"https?://[^/]+(/.+?)(?:\?|#|$)", url)
    return m.group(1) if m else ""


def _keyword_density(content: str, keywords: tuple, window: int = 3000) -> int:
    """Count how many distinct keywords appear in the first `window` chars."""
    sample = content[:window]
    return sum(1 for kw in keywords if kw in sample)


# ---------------------------------------------------------------------------
# Junk signals
# ---------------------------------------------------------------------------
JUNK_MIN_BYTES = 450

JUNK_EXACT = {
    "activities.md", "activity-duration.md", "blog.md", "book-a-trip.md",
    "check-list.md", "contact-us.md", "destination.md", "destinations.md",
    "enquiry-thank-you-page.md", "expedition.md", "guide-porter.md",
    "gwt-test.md", "gwttttttt.md", "itinerary-template.md", "khhj.md",
    "my-account.md", "my-account-2.md", "newsletter.md", "online-payment.md",
    "other.md", "our-e-brochure.md", "payment-info.md", "peak-climbing.md",
    "premium-packages.md", "sasd.md", "search-result.md", "shivaa.md",
    "thank-you.md", "tours.md", "tours-2.md", "travellers-information.md",
    "trekking-equipment.md", "trekking-grades.md", "trekking-permit-cost.md",
    "trip-search-result.md", "trip-types.md", "wishlist.md",
    "e.md", "e-2.md", "noyal.md", "luca.md", "liz.md", "mike.md",
    "genesis-test.md", "genesis-test-2.md", "testing-genesis.md",
    "sherpahimalaya.genesiswtech.com.md",
}

JUNK_URL_SEGS = (
    "/reveal_cost/", "/my-account", "/wishlist", "/newsletter",
    "/enquiry-thank-you", "/thank-you", "/search-result",
    "/trip-search-result", "/online-payment", "/book-a-trip",
    "/our-e-brochure",
)

JUNK_SLUG_RE = re.compile(
    r"^(tsslauep|gwt|gwttttttt|sasd|khhj|megoqel|wvjvin|fehrger"
    r"|genesis-test|testing-genesis|noyal|luca|shivaa).*\.md$"
)

# ---------------------------------------------------------------------------
# Team signals
# ---------------------------------------------------------------------------
TEAM_URL_SEG = "/team/"

KNOWN_GUIDE_NAMES = {
    "ang-sona-sherpa", "ang-tsering-sherpa", "ang-tsering-sherpa-2",
    "ang-tsering-sherpa-3", "darinje-sherpa", "dawa-rinji-sherpa",
    "doma-sherpa", "lopsang-sherpa", "mingma-chhiri-sherpa",
    "nima-dorji-sherpa", "nima-tashi-sherpa", "rinji-sherpa",
    "sangey-sherpa", "tsering-sherpa",
}

TEAM_CONTENT_SIGNALS = (
    "years of experience", "mountain guide", "trekking guide", "sirdar",
    "born in", "grew up in", "joined sherpa himalaya",
    "certified guide", "nma certified",
)

# ---------------------------------------------------------------------------
# Testimonial signals
# ---------------------------------------------------------------------------
TESTIMONIAL_EXACT = {"testimonials.md"}

TESTIMONIAL_CONTENT = (
    "our guide was", "highly recommend", "amazing experience",
    "5 star", "five star", "great trek", "would recommend",
    "our trip was", "satisfied with", "wonderful experience",
    "exceeded our expectations", "customer review",
)

TESTIMONIAL_KNOWN = {"bijay.md", "cruz-vinson.md", "kelly-jimenez.md"}

# ---------------------------------------------------------------------------
# Company signals
# ---------------------------------------------------------------------------
COMPANY_EXACT = {
    "about-us.md", "why-sherpa-himalaya.md", "why-sherpa.md",
    "legal-documents.md", "terms-condition.md", "our-team.md",
    "why-trekking-company-in-nepal.md",
}
COMPANY_URL_SEGS = (
    "/about-us", "/why-sherpa", "/legal", "/terms",
    "/our-team", "/why-trekking-company",
)
COMPANY_CONTENT = (
    "sherpa himalaya treks & expedition pvt", "our mission",
    "founded by", "trekking agencies' association of nepal",
    "commitment to safety", "liability waiver",
    "terms and conditions", "privacy policy",
)

# ---------------------------------------------------------------------------
# Blog signals
# ---------------------------------------------------------------------------
BLOG_EXACT = {
    "covid-19-guidelines-nepal.md",
    "covid-19-nepal-latest-update-2021.md",
    "covid-19-tourist-protocol-for-nepal.md",
    "nepal-travel-protocol-covid-19.md",
    "nepals-latest-covid-19-protocol-update-noquarantine.md",
    "no-transport-in-khumbu-region-how-people-are-facing-covid-19.md",
    "the-new-height-of-mt-everest-mt-everest-increased-by-2ft.md",
    "new-travel-rules-in-nepal-guides-mandatory-for-trekking-except-in-everest-region.md",
    "altitude-sickness.md", "trekking-season.md",
    "trekking-in-nepal.md", "trekking-region-in-nepal.md",
}
BLOG_URL_SEGS = ("/blog/", "/news/", "/updates/")
BLOG_CONTENT = (
    "covid", "quarantine", "pcr test", "latest update",
    "new regulation", "ntb has announced",
    "mountain sickness", "altitude sickness",
    "trekking season", "autumn season", "spring season",
    "travel advisory",
)
BLOG_DATE_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august"
    r"|september|october|november|december)\s+\d{1,2},\s+20\d{2}\b"
)

# ---------------------------------------------------------------------------
# Information signals
# ---------------------------------------------------------------------------
INFO_EXACT = {
    "nepal-visa-info.md", "what-is-tims-card.md", "how-to-book.md",
    "trekking-faqs.md", "nepal.md", "bhutan.md", "tibet.md",
    "camping-trekking.md", "paragliding-in-pokhara.md",
    "chisapani-nagarkot-hiking.md", "pokhara-city-tour-2.md",
    "everest-day-tour.md",
}
INFO_URL_SEGS = (
    "/nepal-visa-info", "/what-is-tims", "/how-to-book",
    "/faq", "/trekking-faqs", "/trekking-grades",
)
INFO_FILENAME_KW = (
    "faq", "visa", "tims", "how-to", "permit", "policy",
    "insurance", "payment", "checklist",
)
INFO_CONTENT = (
    "how to apply", "visa on arrival", "immigration",
    "tims card", "permit fee", "trekking permit",
    "frequently asked", "step 1", "step 2", "step 3",
    "how to book", "what is",
)

# ---------------------------------------------------------------------------
# Expedition signals
# ---------------------------------------------------------------------------
EXPEDITION_URL_SEGS = ("/expedition/",)
EXPEDITION_FILE_KW  = ("expedition",)
EXPEDITION_CONTENT  = (
    "base camp climbing", "high camp", "summit push",
    "liaison officer", "garbage deposit", "climbing permit",
    "8000m", "8,000m", "mountaineering expedition",
    "advanced base camp", "rotation climb",
)

# ---------------------------------------------------------------------------
# Peak signals
# ---------------------------------------------------------------------------
PEAK_URL_SEGS = ("/peak-climbing/", "/peaks/")
PEAK_FILE_RE  = re.compile(
    r"(island|mera|lobuche|pisang|yala|thorong|pokalde|naya-khanga|pachermo"
    r"|paldor|dhampus|larke|singu-chuli|tent|chulu|saribung|himlung"
    r"|amadablam|baruntse|pumori|tukuche).*\.md$"
)
PEAK_CONTENT = (
    "peak climbing", "technical climb", "fixed rope",
    "crampon", "ice axe", "climbing permit for",
    "peak permit", "summit register",
)

# ---------------------------------------------------------------------------
# Trek signals
# ---------------------------------------------------------------------------
TREK_URL_SEGS = ("/trekking/", "/trek/", "/treks/")
TREK_FILE_KW  = (
    "-trek", "ebc-trek", "ebc-with", "phaplu-to-everest",
    "sherpani-col", "rolwaling", "dolpa-circuit", "tsum-valley",
    "langtang", "manaslu-circuit", "kanchenjunga-circuit",
    "ghorepani", "helambu", "tamang-heritage", "nar-phu",
    "makalu-base-camp", "numbur", "panch-pokhari", "jumla-rara",
    "arun-valley", "pikey", "shortest-trek",
)
TREK_CONTENT = (
    "day 01: arrival", "day 1: arrival",
    "tea house", "teahouse", "trekking trail",
    "overnight at", "trek to", "hike to",
    "acclimatization day",
)

# ---------------------------------------------------------------------------
# Tour signals
# ---------------------------------------------------------------------------
TOUR_URL_SEGS = ("/tours/", "/tour/", "/activities/")
TOUR_FILE_KW  = (
    "safari", "jungle", "-tour", "overland", "paragliding",
    "rafting", "helicopter", "city-tour", "day-tour",
    "pokhara", "bhutan", "tibet", "india", "mustang-overland", "camping",
)
TOUR_CONTENT = (
    "jungle safari", "elephant ride", "canoeing", "bird watching",
    "helicopter tour", "city tour", "overland tour",
    "day trip", "full day tour", "paragliding",
)

# ---------------------------------------------------------------------------
# Region signals
# ---------------------------------------------------------------------------
REGION_FILE_KW  = ("-region",)
REGION_URL_SEGS = ("/destinations/",)
REGION_CONTENT  = (
    "trips available", "region overview",
    "trekking in the region", "explore the region",
)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(filepath: Path) -> str:
    """Return a CATEGORIES string for the given markdown file."""
    raw, content = _read(filepath)
    name       = filepath.name.lower()
    stem       = filepath.stem.lower()
    url        = _extract_url(raw)
    path       = _url_path(url)
    file_bytes = filepath.stat().st_size

    # -- 1. JUNK -------------------------------------------------------------
    if file_bytes < JUNK_MIN_BYTES:
        return "junk"
    if name in JUNK_EXACT:
        return "junk"
    if JUNK_SLUG_RE.match(name):
        return "junk"
    if any(seg in path for seg in JUNK_URL_SEGS):
        return "junk"

    # -- 2. TEAM -------------------------------------------------------------
    if TEAM_URL_SEG in path:
        return "team"
    if stem in KNOWN_GUIDE_NAMES:
        return "team"
    if name.endswith("-sherpa.md") and name not in COMPANY_EXACT:
        return "team"
    if _keyword_density(content, TEAM_CONTENT_SIGNALS) >= 2:
        return "team"

    # -- 3. TESTIMONIALS -----------------------------------------------------
    if name in TESTIMONIAL_EXACT or name in TESTIMONIAL_KNOWN:
        return "testimonials"
    if "testimonial" in name or "review" in name or "/testimonial" in path:
        if any(sig in content for sig in TESTIMONIAL_CONTENT):
            return "testimonials"

    # -- 4. COMPANY ----------------------------------------------------------
    if name in COMPANY_EXACT:
        return "company"
    if any(seg in path for seg in COMPANY_URL_SEGS):
        return "company"
    if _keyword_density(content, COMPANY_CONTENT) >= 2:
        return "company"

    # -- 5. BLOGS (exact name / URL) -----------------------------------------
    if name in BLOG_EXACT:
        return "blogs"
    if any(seg in path for seg in BLOG_URL_SEGS):
        return "blogs"

    # -- 6. INFORMATION (exact overrides before blog content signals) ---------
    if name in INFO_EXACT:
        return "information"
    if any(seg in path for seg in INFO_URL_SEGS):
        return "information"
    if any(kw in name for kw in INFO_FILENAME_KW):
        return "information"

    # -- 5b. BLOGS (content signals, run after info exact overrides) ----------
    if _keyword_density(content, BLOG_CONTENT) >= 2:
        return "blogs"
    if BLOG_DATE_RE.search(content) and file_bytes > 1500:
        return "blogs"

    # -- 6b. INFORMATION (content signals fallback) ---------------------------
    if _keyword_density(content, INFO_CONTENT) >= 2:
        return "information"

    # -- 7. DESTINATIONS / EXPEDITIONS ---------------------------------------
    has_exp_kw  = any(kw in name for kw in EXPEDITION_FILE_KW)
    has_exp_url = any(seg in path for seg in EXPEDITION_URL_SEGS)
    has_exp_cnt = _keyword_density(content, EXPEDITION_CONTENT) >= 2

    if has_exp_kw or has_exp_url:
        return "destinations/expeditions"
    if has_exp_cnt and "peak climbing" not in content:
        return "destinations/expeditions"

    # -- 8. DESTINATIONS / PEAKS ---------------------------------------------
    if name.endswith("-peak.md"):
        return "destinations/peaks"
    if PEAK_FILE_RE.search(name) and "expedition" not in name:
        return "destinations/peaks"
    if any(seg in path for seg in PEAK_URL_SEGS):
        return "destinations/peaks"
    if _keyword_density(content, PEAK_CONTENT) >= 2:
        return "destinations/peaks"

    # -- 9. DESTINATIONS / TREKS ---------------------------------------------
    if any(kw in name for kw in TREK_FILE_KW):
        return "destinations/treks"
    if any(seg in path for seg in TREK_URL_SEGS):
        return "destinations/treks"
    if _keyword_density(content, TREK_CONTENT) >= 2:
        return "destinations/treks"

    # -- 10. DESTINATIONS / TOURS --------------------------------------------
    if any(kw in name for kw in TOUR_FILE_KW):
        return "destinations/tours"
    if any(seg in path for seg in TOUR_URL_SEGS):
        return "destinations/tours"
    if _keyword_density(content, TOUR_CONTENT) >= 2:
        return "destinations/tours"

    # -- 11. DESTINATIONS / REGIONS ------------------------------------------
    if any(kw in name for kw in REGION_FILE_KW):
        return "destinations/regions"
    if any(seg in path for seg in REGION_URL_SEGS):
        return "destinations/regions"
    if _keyword_density(content, REGION_CONTENT) >= 1:
        return "destinations/regions"

    # -- FALLBACK ------------------------------------------------------------
    return "junk"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def separate_files(dry_run: bool = False, move: bool = False) -> None:
    """
    Walk SOURCE_DIR, classify every .md, and copy/move it to TARGET_DIR/<category>/.

    Parameters
    ----------
    dry_run : bool  -- Print only; no file I/O.
    move    : bool  -- Move instead of copy (default: copy, keeps originals).
    """
    if not SOURCE_DIR.exists():
        print(f"[ERROR] Source directory not found: {SOURCE_DIR}")
        print("        Run parser.py first to populate data/processed/raw_md/")
        return

    md_files = sorted(SOURCE_DIR.rglob("*.md"))
    if not md_files:
        print("[WARN] No .md files found in source directory.")
        return

    counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    op      = "MOVE" if move else "COPY"
    divider = "-" * 65

    print(divider)
    print(f"  Separator -- {op} mode  {'[DRY RUN]' if dry_run else ''}")
    print(f"  Source : {SOURCE_DIR}")
    print(f"  Target : {TARGET_DIR}")
    print(divider)
    print()

    for fp in md_files:
        cat       = classify(fp)
        counts[cat] = counts.get(cat, 0) + 1
        label     = f"[{cat.upper().replace('/', ' > ')}]"
        dest_dir  = TARGET_DIR / cat
        dest_path = dest_dir / fp.name

        print(f"  {label:<42} {fp.name}")

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(fp), dest_path)
            else:
                shutil.copy2(fp, dest_path)

    # Summary
    total = len(md_files)
    print(f"\n{divider}")
    print(f"  Total files : {total}")
    print(divider)
    for cat in CATEGORIES:
        n = counts.get(cat, 0)
        if n:
            bar = "#" * min(n, 40)
            print(f"  {cat:<35} {n:>4}  {bar}")
    print(divider)

    if dry_run:
        print("  [DRY RUN] No files were written.")
    else:
        print(f"  Files written to: {TARGET_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dry  = "--dry-run" in sys.argv or "-n" in sys.argv
    move = "--move"    in sys.argv or "-m" in sys.argv
    separate_files(dry_run=dry, move=move)
