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
    extended_history_file="extended_history_feed.xml"
):
    """Generates or updates the summary RSS feed file with a new summary entry."""
    logger.info(f"\nGenerating RSS feed entry and updating {output_feed_file}...")

    # Load entries from the extended history file
    all_entries = []
    if os.path.exists(extended_history_file):
        try:
            existing_history_feed = feedparser.parse(extended_history_file)
            if existing_history_feed.bozo:
                logger.warning(f"  Extended history file {extended_history_file} may be malformed. Bozo reason: {existing_history_feed.bozo_exception}. Will attempt to continue.")

            for entry in existing_history_feed.entries:
                all_entries.append(entry)

            logger.info(f"  Loaded {len(existing_history_feed.entries)} entries from extended history file.")
        except Exception as e:
            logger.warning(f"  Warning: Could not load or parse extended history file {extended_history_file}: {e}. Starting with a new history.", exc_info=True)
    
    print(f"  Loaded {len(all_entries)} entries from extended history file.")

    # Add a new summary entry if provided
    if combined_summary:
        new_entry = {
            'title': f"Summary for {datetime.date.today().strftime('%Y-%m-%d')}",
            'id': f"urn:summary:{datetime.date.today().strftime('%Y%m%d')}:{output_feed_title.replace(' ', '_')}",
            'published': datetime.datetime.now(datetime.timezone.utc),
            'updated': datetime.datetime.now(datetime.timezone.utc),
            'link': output_feed_link,
            'content': combined_summary
        }
        all_entries.insert(0, new_entry)  # Add new entry to the top

    # Create FeedGenerator for output feed first (to match test mock side_effect order)
    fg = FeedGenerator()
    fg.title(output_feed_title)
    fg.link(href=output_feed_link, rel='alternate')
    fg.description(output_feed_description)
    fg.language('en')

    # Create FeedGenerator for extended history
    history_fg = FeedGenerator()
    history_fg.title(output_feed_title)
    history_fg.link(href=output_feed_link, rel='alternate')
    history_fg.description(output_feed_description)
    history_fg.language('en')

    # Save all entries to the extended history file
    try:
        for entry in all_entries:
            fe = history_fg.add_entry()
            fe.title(entry.get('title'))
            fe.id(entry.get('id'))
            fe.published(entry.get('published'))
            fe.updated(entry.get('updated'))
            if entry.get('link'):
                fe.link(href=entry.get('link'))
            fe.content(entry.get('content'), type='text')

        history_fg.rss_file(extended_history_file, pretty=True)
        logger.info(f"  Successfully updated extended history file: {extended_history_file}")
    except Exception as e:
        logger.error(f"  Error saving extended history file to {extended_history_file}: {e}")

    # Truncate to at most 10 posts for the output feed
    truncated_entries = all_entries[:10]

    # Add truncated entries to the FeedGenerator object
    for entry in truncated_entries:
        fe = fg.add_entry()
        fe.title(entry.get('title'))
        fe.id(entry.get('id'))
        fe.published(entry.get('published'))
        fe.updated(entry.get('updated'))
        if entry.get('link'):
            fe.link(href=entry.get('link'))
        fe.content(entry.get('content'), type='text')

    # Save the truncated feed
    try:
        fg.rss_file(output_feed_file, pretty=True) # Save as RSS 2.0
        logger.info(f"  Successfully updated RSS feed: {output_feed_file} ({len(truncated_entries)} total entries)")
    except Exception as e:
        logger.error(f"  Error saving RSS feed to {output_feed_file}: {e}")