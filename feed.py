import feedparser
import time

# Import logger and utils
from config import logger
from utils import save_processed_ids
# Import article functions (fetch_article_content might be called within check_feed)
from article import fetch_article_content

def fetch_rss_feed(feed_url):
    """Fetches and parses the RSS feed.
    
    Raises an exception if the feed is invalid or empty."""
    logger.info(f"Fetching RSS feed: {feed_url}")
    feed = feedparser.parse(feed_url)
    if feed.bozo:
        logger.warning(f"RSS feed fetch exception. Bozo reason: {feed.bozo_exception}")
        raise Exception(f"RSS feed fetch exception. Bozo reason: {feed.bozo_exception}")
    if not feed.entries:
        logger.warning("No entries found in the feed.")
        raise Exception("No entries found in the feed.")
    return feed.entries

def check_feed(feed_url, use_feed_summary, processed_ids):
    """Checks the feed for new articles and returns a list of new entries with their content.

    Args:
        feed_url (str): The URL of the RSS feed.
        use_feed_summary (bool): If True, use summary from feed entry, else fetch full content.
        processed_ids (set): A set of already processed article IDs.

    Returns:
        tuple: A tuple containing:
            - list: A list of tuples, where each inner tuple is (entry, content_to_summarize).
            - set: The updated set of processed article IDs.

    Raises:
        Exception: If the feed is invalid, empty, or cannot be read.
    """
    logger.info(f"Checking feed: {feed_url}")
    entries = fetch_rss_feed(feed_url)
    new_entries_found = [] # Local buffer for new entries found in this run
    updated_processed_ids = processed_ids.copy() # Work on a copy
    ids_to_save = set() # Track IDs that need saving for this run

    for entry in entries:
        # Determine a unique ID for the article
        article_id = entry.get('guid') or entry.get('id') or entry.get('link')
        if not article_id:
            logger.warning(f"  Skipping entry with no guid, id, or link: {entry.get('title', 'No Title')}")
            continue

        if article_id not in updated_processed_ids:
            title = entry.get('title', 'No Title')
            link = entry.get('link')
            logger.info(f"  Found new article: {title}")
            # Mark as potentially processed NOW to handle potential errors later
            updated_processed_ids.add(article_id)
            ids_to_save.add(article_id) # Mark this ID as needing to be saved

            content_to_summarize = None
            if use_feed_summary:
                logger.info("    Using summary/description from feed entry.")
                content_to_summarize = entry.get('summary') or entry.get('description')
                if not content_to_summarize:
                     logger.warning("    No summary/description found in feed entry.")
            elif link:
                 # Fetch content immediately if not using feed summary
                 content_to_summarize = fetch_article_content(link)
                 # Delay is handled inside fetch_article_content now
                 if not content_to_summarize:
                      logger.warning(f"    Failed to fetch content for '{title}'. Will use title only if needed.")
                      # Optionally provide title as fallback content?
                      # content_to_summarize = f"Title: {title}\n(Content could not be fetched)"
            else:
                 logger.warning(f"    No link found for '{title}' and not using feed summary.")

            # Buffer the entry and the fetched/extracted content (even if None)
            new_entries_found.append((entry, content_to_summarize))
        # else: # Log skipped articles at debug level if desired
            # logger.debug(f"  Skipping already processed article: {entry.get('title', 'No Title')}")

    logger.info(f"Finished checking feed. Found {len(new_entries_found)} new articles.")
    if ids_to_save: # Only save if new IDs were actually added in this run
         save_processed_ids(updated_processed_ids) # Save the updated full set

    # Return the new entries found and the updated set of processed IDs for this run
    return new_entries_found, updated_processed_ids 