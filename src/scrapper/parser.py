import json
import os
import time
import requests
import trafilatura

# Load the list you provided (assuming you saved it as data/raw/urls.json)
URLS_FILE = "data/raw/urls.json"

def run_deep_scraper():
    if not os.path.exists(URLS_FILE):
        print("❌ urls.json not found! Please run crawler or save the list first.")
        return

    with open(URLS_FILE, "r") as f:
        urls = json.load(f)

    print(f"🚀 Processing {len(urls)} links...")

    for url in urls:
        # 1. SKIP IMAGES and SYSTEM PAGES
        if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.pdf']):
            continue
        if any(x in url for x in ["/wp-content/", "author/", "cart/", "checkout/"]):
            continue

        try:
            # 2. DECIDE FOLDER (Scalability)
            folder = "destinations"
            if "/blog/" in url or "covid-19" in url: folder = "blogs"
            elif any(x in url for x in ["about-us", "terms", "legal"]): folder = "company"
            elif "faq" in url: folder = "insights"
            elif "testimonial" in url: folder = "insights"

            # 3. SCRAPE CONTENT
            # trafilatura is great because it extracts tables (Itineraries/Gears) perfectly
            downloaded = trafilatura.fetch_url(url)
            content = trafilatura.extract(downloaded, include_tables=True)

            if content:
                # Create a safe filename from the URL slug
                slug = url.strip("/").split("/")[-1]
                save_dir = f"data/processed/{folder}"
                os.makedirs(save_dir, exist_ok=True)
                
                with open(f"{save_dir}/{slug}.md", "w", encoding="utf-8") as f:
                    f.write(f"# URL: {url}\n\n{content}")
                
                print(f"💾 Processed: {folder}/{slug}.md")
            
            # Wait 1 second to avoid getting blocked by the website
            time.sleep(1)

        except Exception as e:
            print(f"⚠️ Error on {url}: {e}")

if __name__ == "__main__":
    run_deep_scraper()