import requests
import time
from bs4 import BeautifulSoup

# Import logger and constants from config
from config import logger, USER_AGENT, REQUEST_TIMEOUT_SECONDS, REQUEST_DELAY_SECONDS

def clean_text(text):
    """Clean extracted text by removing extra whitespace and normalizing newlines."""
    if not text:
        return text
    # Remove extra whitespace and normalize newlines
    cleaned = " ".join(text.split())
    # Basic punctuation spacing correction
    cleaned = cleaned.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
    # Could add more sophisticated cleaning here if needed
    return cleaned

def fetch_article_content(url):
    """Fetches and extracts plain text content from an article URL."""
    logger.info(f"  Fetching article content: {url}")
    try:
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.content, 'html.parser')

        # --- Content Extraction (Heuristic - might need adjustment per site) ---
        # 1. Try common article tags
        article_body = soup.find('article') or \
                      soup.find('div', class_=lambda x: x and 'article' in x.lower()) or \
                      soup.find('div', id=lambda x: x and 'content' in x.lower()) or \
                      soup.find('main')

        if article_body:
            # Prioritize paragraphs, then list items, then headings within the main body
            text_elements = article_body.find_all(['p', 'li', 'h1', 'h2', 'h3'])
        else:
            # 2. Fallback: Get all paragraphs from the body
            logger.warning("  Specific article container not found, falling back to all <p> tags.")
            text_elements = soup.find_all('p')

        if not text_elements:
            logger.warning(f"  No text paragraphs found for {url}")
            return None

        # Join text, preserving some structure by adding newlines between elements
        content = "\n".join(p.get_text().strip() for p in text_elements if p.get_text().strip())
        cleaned_content = clean_text(content) # Clean the final joined text

        # Add a small delay after successful fetch
        time.sleep(REQUEST_DELAY_SECONDS)

        return cleaned_content

    except requests.exceptions.RequestException as e:
        logger.error(f"  Error fetching article {url}: {e}")
        # Add delay even on fetch failure to avoid overwhelming the server
        time.sleep(REQUEST_DELAY_SECONDS)
        return None
    except Exception as e:
        logger.error(f"  Error parsing article {url}: {e}")
        # Add delay on other errors too
        time.sleep(REQUEST_DELAY_SECONDS)
        return None 