import requests
import re
from bs4 import BeautifulSoup


def scrape_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join([para.get_text() for para in paragraphs])
        content = re.sub(r"\s+", " ", content)
        return content.strip()
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


with open("links.txt") as f:
    counter = 0
    for line in f:
        url = line.strip()
        with open(f"knowledge-base/input-{counter}.txt", "w") as f2:
            print(f"{counter} done")
            f2.write(str(scrape_content(url)))
        counter += 1
