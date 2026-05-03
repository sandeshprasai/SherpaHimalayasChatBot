"""
parser.py
---------
Incrementally scrapes pages discovered by crawller.py.

Workflow
────────
1. Load `data/raw/urls.json`  (list of {"url", "lastmod"} dicts)
2. Load `data/raw/scrape_state.json`  (persistent ledger: {url -> lastmod_scraped})
3. For each URL:
      - Skip if url already in ledger AND lastmod has not changed.
      - Otherwise scrape with trafilatura.
4. Save scraped content to `data/processed/raw_md/<slug>.md`
5. Update the ledger so next run skips this URL unless it changes again.

All physical categorisation is delegated to `src/utills/separator.py`.
"""

import json
import os
import time
import re

import trafilatura

# ─── Paths ────────────────────────────────────────────────────────────────────
URLS_FILE   = "data/raw/urls.json"
STATE_FILE  = "data/raw/scrape_state.json"
OUTPUT_DIR  = "data/processed/raw_md"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_json(path: str, default):
    """Load JSON file, return default if missing or corrupt."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _url_to_slug(url: str) -> str:
    """Convert a URL to a safe filesystem slug."""
    slug = url.rstrip("/").split("/")[-1]
    # Fallback for root-level URLs (e.g. homepage)
    if not slug:
        slug = re.sub(r"[^\w-]", "-", url)
    # Strip query strings
    slug = slug.split("?")[0].split("#")[0]
    return slug or "index"


# ─── Main runner ──────────────────────────────────────────────────────────────

def run_incremental_scraper(force: bool = False, delay: float = 1.0) -> None:
    """
    Parameters
    ----------
    force : bool
        If True, re-scrape every URL regardless of lastmod.
    delay : float
        Seconds to wait between requests (be polite to the server).
    """
    # 1. Load URL list
    url_entries = _load_json(URLS_FILE, [])
    if not url_entries:
        print(f"[ERROR] {URLS_FILE} not found or empty. Run crawller.py first.")
        return

    # 2. Load persistent state ledger
    state: dict[str, str | None] = _load_json(STATE_FILE, {})

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total      = len(url_entries)
    skipped    = 0
    scraped    = 0
    failed     = 0

    print(f"[START] Processing {total} URLs  (force={force})\n")

    for entry in url_entries:
        url     = entry.get("url", "")
        lastmod = entry.get("lastmod")   # may be None

        if not url:
            continue

        # 3. Incremental check
        if not force and url in state:
            stored_lastmod = state[url]
            # Skip if lastmod has NOT changed (or neither run has a lastmod)
            if stored_lastmod == lastmod:
                skipped += 1
                continue

        slug = _url_to_slug(url)
        out_path = os.path.join(OUTPUT_DIR, f"{slug}.md")

        try:
            # 4. Scrape
            downloaded = trafilatura.fetch_url(url)
            content    = trafilatura.extract(
                downloaded,
                include_tables=True,
                include_links=False,
                include_comments=False,
                no_fallback=False,
            )

            if content and len(content.strip()) > 80:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(f"# URL: {url}\n")
                    if lastmod:
                        f.write(f"# LASTMOD: {lastmod}\n")
                    f.write(f"\n{content.strip()}\n")

                print(f"  [OK]  [{scraped + 1}] Scraped -> {slug}.md")
                scraped += 1

                # 5. Update ledger ONLY on success
                state[url] = lastmod

            else:
                print(f"  [SKIP] Skipped (empty/stub) -> {url}")
                # Still update ledger so we don't re-attempt on every run
                state[url] = lastmod

        except Exception as e:
            print(f"  [ERR]  Error on {url}: {e}")
            failed += 1

        time.sleep(delay)

    # Persist updated ledger
    _save_json(STATE_FILE, state)

    print(f"""
{'─' * 55}
  Total URLs     : {total}
  Scraped (new)  : {scraped}
  Skipped (same) : {skipped}
  Failed         : {failed}
{'─' * 55}
  State saved to : {STATE_FILE}
  Files saved to : {OUTPUT_DIR}
""")


if __name__ == "__main__":
    import sys
    force_flag = "--force" in sys.argv or "-f" in sys.argv
    run_incremental_scraper(force=force_flag)