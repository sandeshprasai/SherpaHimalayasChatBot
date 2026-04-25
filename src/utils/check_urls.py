import json

data = json.load(open("data/raw/urls.json"))
print(f"Total URLs : {len(data)}")
print()
for e in data[:8]:
    print(f"  lastmod={e['lastmod']}  |  {e['url']}")
