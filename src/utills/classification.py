"""
classification.py
-----------------
Classifies processed markdown files from data/processed/** into a structured
taxonomy under data/separated/.

Output layout
─────────────
data/separated/
├── destinations/
│   ├── treks/          Trek itineraries
│   ├── expeditions/    Mountain expeditions (8000m peaks, technical climbs)
│   ├── peaks/          Peak-climbing packages
│   ├── tours/          Day-tours, safaris, overland tours, adventure activities
│   └── regions/        Regional overview / listing pages
├── blogs/              News articles, travel updates, informational blog posts
├── information/        Practical travel info (visa, permits, gear, FAQs, etc.)
├── company/            About us, why-us, legal, mission
├── team/               Individual staff / guide profiles
├── testimonials/       Client reviews
└── junk/               Stub pages, test entries, revenue-cost forms, near-empty files
"""

import os
import re
import shutil

# ──────────────────────────── Paths ────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/processed"))
TARGET_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/separated"))

# ──────────────────────────── Category registry ────────────────────────────────

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

for cat in CATEGORIES:
    os.makedirs(os.path.join(TARGET_DIR, cat), exist_ok=True)


# ──────────────────────────── Helpers ──────────────────────────────────────────

def read_file(filepath: str) -> tuple[str, str]:
    """Return (raw_content, lowercase_content)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
        return raw, raw.lower()
    except Exception:
        return "", ""


def extract_url(content: str) -> str:
    """Pull the URL from the first line of the markdown file."""
    match = re.search(r"#\s*URL:\s*(https?://\S+)", content)
    return match.group(1).lower() if match else ""


def url_path(url: str) -> str:
    """Return the path portion of the URL (without query/fragment)."""
    match = re.match(r"https?://[^/]+(/.+?)(?:\?|#|$)", url)
    return match.group(1) if match else ""


# ──────────────────────────── Classification rules ─────────────────────────────
#
# Priority order (highest → lowest):
#   1. JUNK   – stub pages, test entries, form pages, near-empty files
#   2. TEAM   – individual guide / staff profile pages
#   3. TESTIMONIALS – client reviews
#   4. COMPANY – about, legal, why-us, our-team (listing page), contact
#   5. BLOGS  – dated articles: COVID updates, news, altitude info, etc.
#   6. INFORMATION – practical how-to: visa, permits, gear, FAQs, payment
#   7. DESTINATIONS/EXPEDITIONS
#   8. DESTINATIONS/PEAKS
#   9. DESTINATIONS/TREKS
#  10. DESTINATIONS/TOURS
#  11. DESTINATIONS/REGIONS
#  (fallback → junk)

# ── 1. JUNK ── Filenames that are clearly useless stubs / test / form pages ──
JUNK_FILENAME_EXACT = {
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
    # test / random noise pages
    "e.md", "e-2.md", "noyal.md", "luca.md", "liz.md", "mike.md",
    "genesis-test.md", "genesis-test-2.md", "testing-genesis.md",
    "sherpahimalaya.genesiswtech.com.md",
}

JUNK_FILENAME_PREFIX = (
    "tsslauep",            # auto-generated test slugs
)

JUNK_URL_SEGMENTS = (
    "/reveal_cost/",       # cost-reveal form submissions
    "/my-account",
    "/wishlist",
    "/newsletter",
    "/enquiry-thank-you",
    "/thank-you",
    "/search-result",
    "/trip-search-result",
    "/online-payment",
    "/book-a-trip",
    "/our-e-brochure",
)

# Random-name patterns (client form submissions like "jolaine-mooney-2.md")
NAME_NOISE_RE = re.compile(
    r"^[a-z]+-[a-z]+(-(2|3|4|5|6|7|8|9|10|11))?\.md$"
)

JUNK_URL_RE = re.compile(
    r"/(reveal_cost|tsslauep|gwt|genesis-test|gwttttttt|sasd|khhj|shivaa|noyal|luca)")
    # Extended below to avoid over-long regexp literal
JUNK_SLUG_PATTERNS = re.compile(
    r"^(tsslauep|gwt|gwttttttt|sasd|khhj|megoqelvzlhjpqtrwrqmpowy|wvjvinktqyoeoogwr"
    r"|fehrger|genesis-test|testing-genesis).*\.md$"
)

# ── 2. TEAM – individual profile pages ──────────────────────────────────────
TEAM_URL_SEGMENT = "/team/"

KNOWN_GUIDE_NAMES = {
    "ang-sona-sherpa", "ang-tsering-sherpa", "ang-tsering-sherpa-2",
    "ang-tsering-sherpa-3", "darinje-sherpa", "dawa-rinji-sherpa",
    "doma-sherpa", "lopsang-sherpa", "mingma-chhiri-sherpa",
    "nima-dorji-sherpa", "nima-tashi-sherpa", "rinji-sherpa",
    "sangey-sherpa", "tsering-sherpa",
}

# ── 3. TESTIMONIALS ──────────────────────────────────────────────────────────
TESTIMONIAL_FILENAMES = {"testimonials.md"}
TESTIMONIAL_CONTENT_SIGNALS = (
    "testimonial", "our guide was", "highly recommend", "amazing experience",
    "5 star", "five star", "great trek", "would recommend",
    "our trip was", "satisfied with", "wonderful experience",
)

# ── 4. COMPANY ───────────────────────────────────────────────────────────────
COMPANY_FILENAME_EXACT = {
    "about-us.md", "why-sherpa-himalaya.md", "why-sherpa.md",
    "legal-documents.md", "terms-condition.md", "our-team.md",
    "why-trekking-company-in-nepal.md",
}
COMPANY_URL_SEGMENTS = (
    "/about-us", "/why-sherpa", "/legal", "/terms",
    "/our-team", "/why-trekking-company",
)
COMPANY_CONTENT_SIGNALS = (
    "sherpa himalaya treks & expedition pvt", "our mission",
    "founded by", "trekking agencies' association of nepal",
    "commitment to safety",
)

# ── 5. BLOGS (dated articles / news) ─────────────────────────────────────────
BLOG_FILENAME_EXACT = {
    "covid-19-guidelines-nepal.md",
    "covid-19-nepal-latest-update-2021.md",
    "covid-19-tourist-protocol-for-nepal.md",
    "nepal-travel-protocol-covid-19.md",
    "nepals-latest-covid-19-protocol-update-noquarantine.md",
    "no-transport-in-khumbu-region-how-people-are-facing-covid-19.md",
    "the-new-height-of-mt-everest-mt-everest-increased-by-2ft.md",
    "new-travel-rules-in-nepal-guides-mandatory-for-trekking-except-in-everest-region.md",
    "altitude-sickness.md",
    "trekking-season.md",
    "trekking-in-nepal.md",
    "trekking-region-in-nepal.md",
}

BLOG_URL_SEGMENTS = ("/blog/", "/news/", "/updates/")

BLOG_CONTENT_SIGNALS = (
    "covid", "quarantine", "pcr test", "latest update",
    "new regulation", "ntb has announced", "nepal tourism board (ntb) has announced",
    "mountain sickness", "altitude sickness",
    "trekking season", "autumn season", "spring season",
)

BLOG_DATE_RE = re.compile(r"\b(january|february|march|april|may|june|july|august"
                           r"|september|october|november|december)\s+\d{1,2},\s+20\d{2}\b")

# ── 6. INFORMATION (practical/how-to content) ────────────────────────────────
INFO_FILENAME_EXACT = {
    "nepal-visa-info.md", "what-is-tims-card.md", "how-to-book.md",
    "trekking-faqs.md", "nepal.md", "bhutan.md", "tibet.md",
    "camping-trekking.md", "paragliding-in-pokhara.md",
    "chisapani-nagarkot-hiking.md", "pokhara-city-tour-2.md",
    "everest-day-tour.md",
}

INFO_URL_SEGMENTS = (
    "/nepal-visa-info", "/what-is-tims", "/how-to-book",
    "/faq", "/trekking-faqs", "/trekking-grades",
)

INFO_FILENAME_KEYWORDS = (
    "faq", "visa", "tims", "how-to", "permit", "policy",
    "insurance", "payment", "checklist",
)

INFO_CONTENT_SIGNALS = (
    "how to apply", "visa on arrival", "immigration",
    "tims card", "permit fee", "trekking permit",
    "what is", "how to book", "frequently asked",
    "step 1", "step 2", "step 3",
)

# ── 7-9. DESTINATIONS/EXPEDITIONS, PEAKS, TREKS ──────────────────────────────
EXPEDITION_URL_SEGMENTS = ("/expedition/",)
EXPEDITION_FILE_KEYWORDS = ("expedition",)
EXPEDITION_CONTENT_SIGNALS = (
    "expedition", "base camp climbing", "high camp", "summit push",
    "liaison officer", "garbage deposit", "climbing permit",
    "8000m", "8,000m", "mountaineering expedition",
)

PEAK_URL_SEGMENTS = ("/peak-climbing/", "/peaks/")
PEAK_FILE_KEYWORDS = (
    "-peak", "island-peak", "mera-peak", "lobuche-east",
    "pisang-peak", "yala-peak", "thorong-peak", "pokalde-peak",
    "naya-khanga", "pachermo", "paldor", "dhampus", "larke",
    "singu-chuli", "tent-peak", "chulu", "saribung",
)
PEAK_CONTENT_SIGNALS = (
    "peak climbing", "technical climb", "fixed rope",
    "crampon", "ice axe", "summit", "climbing permit for",
    "peak permit",
)
# Peak filenames end with "-peak.md" or match known peak names
PEAK_FILE_RE = re.compile(
    r"(island|mera|lobuche|pisang|yala|thorong|pokalde|naya-khanga|pachermo"
    r"|paldor|dhampus|larke|singu-chuli|tent|chulu|saribung|himlung"
    r"|amadablam|baruntse|pumori|tukuche|himlung).*\.md$"
)

TREK_URL_SEGMENTS = ("/trekking/", "/trek/", "/treks/")
TREK_FILE_KEYWORDS = (
    "-trek", "ebc-trek", "ebc-with", "phaplu-to-everest",
    "sherpani-col", "rolwaling", "dolpa-circuit", "tsum-valley",
    "langtang", "manaslu-circuit", "kanchenjunga-circuit",
    "ghorepani", "helambu", "tamang-heritage", "nar-phu",
    "makalu-base-camp", "numbur", "panch-pokhari", "jumla-rara",
    "arun-valley", "pikey", "shortest-trek",
)
TREK_CONTENT_SIGNALS = (
    "day 01: arrival", "day 1: arrival",
    "tea house", "teahouse", "trekking trail",
    "overnight at", "trek to", "hike to",
    "trekking permit", "tims card",
    "acclimatization day",
)

# ── 10-11. DESTINATIONS/TOURS & REGIONS ──────────────────────────────────────
TOUR_FILE_KEYWORDS = (
    "safari", "jungle", "tour", "overland", "paragliding", "rafting",
    "helicopter", "city-tour", "day-tour", "pokhara", "bhutan",
    "tibet", "india", "mustang-overland", "camping",
)
TOUR_URL_SEGMENTS = ("/tours/", "/tour/", "/activities/")
TOUR_CONTENT_SIGNALS = (
    "jungle safari", "elephant ride", "canoeing", "bird watching",
    "helicopter tour", "city tour", "overland tour",
    "day trip", "full day tour", "paragliding",
)

REGION_FILE_KEYWORDS = ("-region",)
REGION_URL_SEGMENTS = ("/destinations/",)
REGION_CONTENT_SIGNALS = (
    "trips available", "trip(s) available", "trips available in",
    "region overview", "trekking in the region",
)


# ──────────────────────────── Main classifier ──────────────────────────────────

def classify_file(filename: str, filepath: str) -> str:
    """
    Returns one of the CATEGORIES strings for the given file.
    Strategy: URL first → filename exact match → filename keyword → content signals.
    """
    raw, content = read_file(filepath)
    name = filename.lower()
    url  = extract_url(raw)
    path = url_path(url)
    filesize = os.path.getsize(filepath)

    # ── 1. JUNK ──────────────────────────────────────────────────────────────

    # Files too small to have real content (< 450 bytes is almost always a stub)
    if filesize < 450:
        return "junk"

    if name in JUNK_FILENAME_EXACT:
        return "junk"

    if JUNK_SLUG_PATTERNS.match(name):
        return "junk"

    if name.startswith(JUNK_FILENAME_PREFIX):
        return "junk"

    if any(seg in path for seg in JUNK_URL_SEGMENTS):
        return "junk"

    # ── 2. TEAM ──────────────────────────────────────────────────────────────

    if TEAM_URL_SEGMENT in path:
        return "team"

    slug = name.replace(".md", "")
    if slug in KNOWN_GUIDE_NAMES:
        return "team"

    # Names like "darinje-sherpa.md", "nima-dorji-sherpa.md"
    if name.endswith("-sherpa.md") and slug not in {
        "ang-sona-sherpa", "ang-tsering-sherpa"
    }:
        return "team"

    # ── 3. TESTIMONIALS ──────────────────────────────────────────────────────

    if name in TESTIMONIAL_FILENAMES:
        return "testimonials"

    if any(sig in content for sig in TESTIMONIAL_CONTENT_SIGNALS):
        # Only if the file is primarily a review (check URL or name)
        if "testimonial" in name or "review" in name or "/testimonial" in path:
            return "testimonials"

    # insight files (bijay.md, cruz-vinson.md, kelly-jimenez.md)
    if any(
        name == n for n in ("bijay.md", "cruz-vinson.md", "kelly-jimenez.md")
    ):
        return "testimonials"

    # ── 4. COMPANY ───────────────────────────────────────────────────────────

    if name in COMPANY_FILENAME_EXACT:
        return "company"

    if any(seg in path for seg in COMPANY_URL_SEGMENTS):
        return "company"

    if any(sig in content for sig in COMPANY_CONTENT_SIGNALS):
        return "company"

    # ── 5. BLOGS ─────────────────────────────────────────────────────────────

    if name in BLOG_FILENAME_EXACT:
        return "blogs"

    if any(seg in path for seg in BLOG_URL_SEGMENTS):
        return "blogs"

    if any(sig in content for sig in BLOG_CONTENT_SIGNALS):
        return "blogs"

    # Articles with a publication date in text body
    if BLOG_DATE_RE.search(content) and filesize > 1500:
        return "blogs"

    # ── 6. INFORMATION ───────────────────────────────────────────────────────

    if name in INFO_FILENAME_EXACT:
        return "information"

    if any(seg in path for seg in INFO_URL_SEGMENTS):
        return "information"

    if any(kw in name for kw in INFO_FILENAME_KEYWORDS):
        return "information"

    if any(sig in content for sig in INFO_CONTENT_SIGNALS):
        return "information"

    # ── 7. DESTINATIONS / EXPEDITIONS ────────────────────────────────────────

    has_expedition_name = any(kw in name for kw in EXPEDITION_FILE_KEYWORDS)
    has_expedition_content = any(sig in content for sig in EXPEDITION_CONTENT_SIGNALS)
    has_expedition_url = any(seg in path for seg in EXPEDITION_URL_SEGMENTS)

    if has_expedition_name or has_expedition_url:
        return "destinations/expeditions"

    # ── 8. DESTINATIONS / PEAKS ──────────────────────────────────────────────

    if name.endswith("-peak.md"):
        return "destinations/peaks"

    if PEAK_FILE_RE.search(name):
        # Distinguish expeditions from peaks: "expedition" = expeditions
        if "expedition" not in name:
            return "destinations/peaks"

    if any(seg in path for seg in PEAK_URL_SEGMENTS):
        return "destinations/peaks"

    if has_expedition_content and "peak climbing" in content:
        return "destinations/peaks"

    # ── 9. DESTINATIONS / TREKS ──────────────────────────────────────────────

    if any(kw in name for kw in TREK_FILE_KEYWORDS):
        return "destinations/treks"

    if any(seg in path for seg in TREK_URL_SEGMENTS):
        return "destinations/treks"

    if any(sig in content for sig in TREK_CONTENT_SIGNALS):
        return "destinations/treks"

    # ── 10. DESTINATIONS / TOURS ──────────────────────────────────────────────

    if any(kw in name for kw in TOUR_FILE_KEYWORDS):
        return "destinations/tours"

    if any(seg in path for seg in TOUR_URL_SEGMENTS):
        return "destinations/tours"

    if any(sig in content for sig in TOUR_CONTENT_SIGNALS):
        return "destinations/tours"

    # ── 11. DESTINATIONS / REGIONS ───────────────────────────────────────────

    if any(kw in name for kw in REGION_FILE_KEYWORDS):
        return "destinations/regions"

    if any(seg in path for seg in REGION_URL_SEGMENTS):
        return "destinations/regions"

    if any(sig in content for sig in REGION_CONTENT_SIGNALS):
        return "destinations/regions"

    # ── FALLBACK ─────────────────────────────────────────────────────────────
    return "junk"


# ──────────────────────────── Driver ───────────────────────────────────────────

def process_files(dry_run: bool = False) -> None:
    """
    Walk SOURCE_DIR recursively, classify every .md file, and copy it
    into the appropriate subdirectory of TARGET_DIR.

    Parameters
    ----------
    dry_run : bool
        If True, only print classifications without copying files.
    """
    counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    total = 0

    for root, _, files in os.walk(SOURCE_DIR):
        for file in sorted(files):
            if not file.endswith(".md"):
                continue

            total += 1
            src_path = os.path.join(root, file)
            category = classify_file(file, src_path)
            counts[category] = counts.get(category, 0) + 1

            label = f"[{category.upper().replace('/', ' > ')}]"
            print(f"  {label:<40} {file}")

            if not dry_run:
                dest_dir  = os.path.join(TARGET_DIR, category)
                dest_path = os.path.join(dest_dir, file)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(src_path, dest_path)

    # Summary
    print("\n" + "─" * 60)
    print(f"  Total files processed : {total}")
    print("─" * 60)
    for cat, n in sorted(counts.items()):
        if n:
            bar = "█" * min(n, 40)
            print(f"  {cat:<35} {n:>4}  {bar}")
    print("─" * 60)
    if dry_run:
        print("  [DRY RUN] No files were copied.")
    else:
        print(f"  Files written to: {TARGET_DIR}")


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv or "-n" in sys.argv
    process_files(dry_run=dry)