"""
crawller.py
-----------
Discovers all URLs from the Sherpa Himalaya XML sitemap hierarchy.

Saves `data/raw/urls.json` as a list of dicts:
    [{"url": "https://...", "lastmod": "2024-03-12"}, ...]

`lastmod` is used downstream by parser.py to skip pages that haven't
changed since the last scrape, avoiding redundant processing.
"""

import json
import os

import requests
from bs4 import BeautifulSoup

MAIN_SITEMAP = "https://sherpahimalaya.genesiswtech.com/sitemap_index.xml"
OUTPUT_FILE  = "data/raw/urls.json"

# Pages we never want to scrape (saves bandwidth + parser effort)
SKIP_URL_FRAGMENTS = (
    "/wp-content/",
    "/author/",
    "/cart/",
    "/checkout/",
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
    "/reveal_cost/",
)

SKIP_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".pdf", ".gif", ".svg")


def _should_skip(url: str) -> bool:
    url_lower = url.lower()
    if any(url_lower.endswith(ext) for ext in SKIP_EXTENSIONS):
        return True
    if any(frag in url_lower for frag in SKIP_URL_FRAGMENTS):
        return True
    return False


def fetch_links() -> list[dict]:
    """
    Crawl sitemap hierarchy and return consolidated list of
    {"url": str, "lastmod": str | None} dicts.
    """
    print(f"[CRAWL] Accessing sitemap index: {MAIN_SITEMAP}")
    all_entries: dict[str, str | None] = {}   # url -> lastmod (de-dup by URL)

    try:
        resp = requests.get(MAIN_SITEMAP, timeout=15)
        resp.raise_for_status()
        root_soup = BeautifulSoup(resp.content, "xml")
        sub_sitemaps = [loc.text.strip() for loc in root_soup.find_all("loc")]

        for sitemap_url in sub_sitemaps:
            print(f"  [SUB]  Sub-sitemap: {sitemap_url}")
            try:
                sub_resp = requests.get(sitemap_url, timeout=15)
                sub_resp.raise_for_status()
                sub_soup = BeautifulSoup(sub_resp.content, "xml")

                for url_tag in sub_soup.find_all("url"):
                    loc_tag     = url_tag.find("loc")
                    lastmod_tag = url_tag.find("lastmod")

                    if loc_tag is None:
                        continue

                    url     = loc_tag.text.strip()
                    lastmod = lastmod_tag.text.strip() if lastmod_tag else None

                    if _should_skip(url):
                        continue

                    # Prefer the most recent lastmod if URL appears in multiple sitemaps
                    if url not in all_entries or lastmod is not None:
                        all_entries[url] = lastmod

            except Exception as e:
                print(f"  [WARN] Failed to fetch sub-sitemap {sitemap_url}: {e}")

    except Exception as e:
        print(f"[ERROR] Error accessing main sitemap: {e}")
        return []

    entries = [{"url": url, "lastmod": lastmod} for url, lastmod in all_entries.items()]

    os.makedirs("data/raw", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Discovery complete -- {len(entries)} URLs saved to {OUTPUT_FILE}")
    return entries


if __name__ == "__main__":
    fetch_links()