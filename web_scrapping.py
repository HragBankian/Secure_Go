import requests
import re
from urllib.parse import urljoin

def get_all_urls(website_url):
    try:
        response = requests.get(website_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {website_url}: {e}")
        return []

    html = response.text

    # Find all absolute URLs (http, https, www)
    url_pattern = re.compile(
        r'((?:https?://|www\.)[^\s"\'<>]+)', re.IGNORECASE
    )
    found_urls = set(match.group(0) for match in url_pattern.finditer(html))

    # Find all relative URLs in href/src attributes
    rel_url_pattern = re.compile(
        r'(?:href|src)=["\']([^"\']+)', re.IGNORECASE
    )
    for match in rel_url_pattern.finditer(html):
        rel_url = match.group(1)
        # Skip if already absolute
        if not rel_url.startswith(('http://', 'https://', 'www.')):
            abs_url = urljoin(website_url, rel_url)
            found_urls.add(abs_url)

    return found_urls

if __name__ == "__main__":
    url = input("Enter website URL: ")
    all_urls = get_all_urls(url)
    print("Found URLs:")
    for link in all_urls:
        print(link) 