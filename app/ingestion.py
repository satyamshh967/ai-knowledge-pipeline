import requests
from bs4 import BeautifulSoup

from app.models import Document


def fetch_webpage(url: str, title: str) -> Document:
    response = requests.get(
        url,
        timeout=10,
        headers={
            "User-Agent": "AI-Knowledge-Pipeline/1.0"
        }
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = soup.get_text(separator=" ", strip=True)

    return Document(
        title=title,
        source=url,
        content=text,
        metadata={
            "content_type": "webpage"
        }
    )