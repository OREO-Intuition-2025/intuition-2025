import requests
import re
from bs4 import BeautifulSoup
import time


def scrape_content(url):
    try:
        # Add headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()  # Raise error for bad status codes
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Try multiple content extraction strategies
        content = ""
        
        # Strategy 1: Look for main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|article|post|entry'))
        if main_content:
            paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'li'])
            content = " ".join([para.get_text() for para in paragraphs])
        
        # Strategy 2: Fallback to all paragraphs
        if not content or len(content) < 200:
            paragraphs = soup.find_all("p")
            content = " ".join([para.get_text() for para in paragraphs])
        
        # Clean up whitespace
        content = re.sub(r"\s+", " ", content)
        content = content.strip()
        
        # Check if content is meaningful (not just error messages)
        if len(content) < 100 or any(phrase in content.lower() for phrase in [
            'page not found', 'does not exist', 'error', 'access denied',
            'cookie settings', 'javascript is disabled', 'please enable'
        ]):
            return ""
        
        return content
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


with open("links.txt") as f:
    counter = 0
    success_count = 0
    for line in f:
        url = line.strip()
        # Skip empty lines and comment lines
        if not url or url.startswith("#"):
            continue
        try:
            content = scrape_content(url)
            if content:  # Only write if we got content
                with open(f"knowledge-base/input-{counter}.txt", "w", encoding="utf-8") as f2:
                    f2.write(content)
                print(f"✓ {counter} done ({len(content)} chars) - {url}")
                success_count += 1
            else:
                print(f"✗ {counter} skipped (no content) - {url}")
            counter += 1
            time.sleep(1)  # Be polite, wait 1 second between requests
        except Exception as e:
            print(f"✗ {counter} error: {e} - {url}")
            counter += 1

print(f"\nCompleted: {success_count}/{counter} URLs successfully scraped")
