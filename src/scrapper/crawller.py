import requests
from bs4 import BeautifulSoup
import json
import os

def fetch_links():
    # The main sitemap index
    main_sitemap = "https://sherpahimalaya.genesiswtech.com/sitemap_index.xml"
    print(f"🔗 Accessing Sitemap: {main_sitemap}")
    
    all_links = []
    try:
        # 1. Get the main sitemap index
        response = requests.get(main_sitemap)
        soup = BeautifulSoup(response.content, 'xml')
        sub_sitemaps = [loc.text for loc in soup.find_all('loc')]

        # 2. Visit each sub-sitemap (posts, pages, treks)
        for sitemap in sub_sitemaps:
            print(f"  📂 Checking sub-sitemap: {sitemap}")
            res = requests.get(sitemap)
            s_soup = BeautifulSoup(res.content, 'xml')
            links = [loc.text for loc in s_soup.find_all('loc')]
            all_links.extend(links)

        # 3. Save to raw data folder
        os.makedirs("data/raw", exist_ok=True)
        with open("data/raw/urls.json", "w") as f:
            json.dump(list(set(all_links)), f, indent=4)
        
        print(f"✅ Discovery Complete. Found {len(all_links)} total links.")
        print(f"📍 Saved to: data/raw/urls.json")

    except Exception as e:
        print(f"❌ Error during crawling: {e}")

if __name__ == "__main__":
    fetch_links()