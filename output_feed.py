import datetime
import os
import feedparser
from feedgen.feed import FeedGenerator

# Import logger from config
from config import logger

def generate_summary_feed(
    output_feed_file,
    output_feed_title,
    output_feed_link,
    output_feed_description,
    combined_summary,
    first_article_link # Used as a fallback link for the summary entry
):
    """Generates or updates the summary RSS feed file with a new summary entry."""
    logger.info(f"\nGenerating RSS feed entry and updating {output_feed_file}...")
    fg = FeedGenerator()

    # Initialize feed metadata (do this every time to potentially update it)
    fg.title(output_feed_title)
    fg.link(href=output_feed_link, rel='alternate')
    fg.description(output_feed_description)
    fg.language('en') # Or make configurable

    # Try loading existing feed *after* setting metadata
    existing_entries_count = 0
    if os.path.exists(output_feed_file):
        try:
            # Parse the existing file using feedparser
            existing_feed = feedparser.parse(output_feed_file)
            if existing_feed.bozo:
                logger.warning(f"  Existing feed file {output_feed_file} may be malformed. Bozo reason: {existing_feed.bozo_exception}. Will attempt to continue.")

            # Add old entries to the new FeedGenerator object
            for entry in existing_feed.entries:
                old_fe = fg.add_entry() # Create a new entry object within our generator
                old_fe.title(entry.get('title'))
                old_fe.id(entry.get('id'))
                # Use link attribute directly if present
                if 'link' in entry:
                     old_fe.link(href=entry.link) # feedparser typically gives 'link' attribute
                # Handle published and updated dates (feedparser stores them as time_struct)
                if entry.get('published_parsed'):
                    # Create timezone-aware datetime objects assuming UTC if no tz info provided by feedparser
                    pub_dt = datetime.datetime(*entry.published_parsed[:6])
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=datetime.timezone.utc)
                    old_fe.published(pub_dt)
                if entry.get('updated_parsed'):
                     upd_dt = datetime.datetime(*entry.updated_parsed[:6])
                     if upd_dt.tzinfo is None:
                         upd_dt = upd_dt.replace(tzinfo=datetime.timezone.utc)
                     old_fe.updated(upd_dt)
                # Handle content (feedparser puts it in a list)
                if entry.get('content'):
                    # Find the most suitable content type (prefer html or text)
                    content_type = 'text' # default
                    content_value = entry.get('summary', '') # fallback to summary
                    for content_item in entry.content:
                        if hasattr(content_item, 'type') and content_item.type in ['text/html', 'html']:
                            content_type = 'html'
                            content_value = content_item.value
                            break
                        elif hasattr(content_item, 'type') and content_item.type in ['text/plain', 'text']:
                            content_type = 'text'
                            content_value = content_item.value
                            # Don't break, prefer html if found later
                    old_fe.content(content_value, type=content_type)
                elif entry.get('summary'): # Use summary if no content
                    old_fe.content(entry.summary, type='text')
                # Copy other potential fields if needed (e.g., author)
                if entry.get('author'):
                    old_fe.author(name=entry.author)

            existing_entries_count = len(existing_feed.entries)
            logger.info(f"  Loaded {existing_entries_count} existing entries from feed file.")
        except Exception as e:
            logger.warning(f"  Warning: Could not load or parse existing feed file {output_feed_file}: {e}. Starting with a new feed.", exc_info=True)
            # Proceed with the initialized fg object (new feed)
            existing_entries_count = 0 # Ensure count is 0 if loading failed
    else:
         logger.info("  No existing feed file found. Creating a new feed.")

    # --- Create a new feed entry for this summary --- Needed only if summary exists
    if combined_summary:
        fe = fg.add_entry(order='prepend') # Add new entries to the top
        summary_title = f"Summary for {datetime.date.today().strftime('%Y-%m-%d')}"
        fe.title(summary_title)
        # Use a unique ID based on the date and title to avoid collisions
        entry_id = f"urn:summary:{datetime.date.today().strftime('%Y%m%d')}:{output_feed_title.replace(' ', '_')}"
        fe.id(entry_id)
        # Use timezone-aware datetime for publishing dates
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        fe.published(now_utc)
        fe.updated(now_utc)
        # Link: Use the feed link or the first article link as a fallback
        entry_link = first_article_link or output_feed_link
        fe.link(href=entry_link)
        # Content: Use the generated summary
        # Ensure summary is not None or empty before setting content
        fe.content(combined_summary or "Summary could not be generated.", type='text') # Assuming summary is plain text
        logger.info(f"  Added new summary entry: '{summary_title}'")
    else:
        logger.info("  No combined summary was provided. Only preserving existing entries.")

    # Save the updated feed
    try:
        fg.rss_file(output_feed_file, pretty=True) # Save as RSS 2.0
        final_entry_count = existing_entries_count + (1 if combined_summary else 0)
        logger.info(f"  Successfully updated RSS feed: {output_feed_file} ({final_entry_count} total entries)")
    except Exception as e:
        logger.error(f"  Error saving RSS feed to {output_feed_file}: {e}") 