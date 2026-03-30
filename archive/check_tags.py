import os, requests

api_key = os.environ.get("GOOGLE_FONTS_API_KEY", "")
url = f"https://www.googleapis.com/webfonts/v1/webfonts?key={api_key}&capability=FAMILY_TAGS"

resp = requests.get(url)
print(resp.text[:1000])