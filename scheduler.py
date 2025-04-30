import schedule
import time
import threading
import datetime
import random
from dataclasses import dataclass, field

# Import necessary components from other modules
from config import logger, CHECK_INTERVAL_MINUTES
from feed import check_feed
from summarization import summarize_text_with_langchain
from output_feed import generate_summary_feed
from utils import save_processed_ids
from langchain_openai import ChatOpenAI

# --- State Management ---

# Define constants for backoff
INITIAL_BACKOFF_MINUTES = CHECK_INTERVAL_MINUTES # Start backoff at the normal check interval
MAX_BACKOFF_HOURS = 24
MAX_BACKOFF_MINUTES = MAX_BACKOFF_HOURS * 60
BACKOFF_FACTOR = 2 # Exponential backoff factor
JITTER_FACTOR = 0.2 # Add +/- 20% jitter to backoff

@dataclass
class FeedState:
    url: str
    use_feed_summary: bool
    last_check_time: datetime.datetime = field(default=datetime.datetime.min)
    next_check_time: datetime.datetime = field(default=datetime.datetime.min) # Placeholder
    failure_count: int = 0
    current_backoff_minutes: float = INITIAL_BACKOFF_MINUTES
    is_checking: bool = False # Flag to prevent concurrent checks for the same feed

class AppState:
    def __init__(self, processed_ids: set, feed_urls: list[str], use_feed_summary: bool, initial_check_interval: int):
        self.processed_ids = processed_ids
        self.articles_buffer = [] # List to store (entry, content_to_summarize, feed_url)
        self.lock = threading.Lock() # Lock for thread-safe access to buffer/ids/feed_states
        # Initialize state for each feed
        now = datetime.datetime.now() # Get current time (respects freeze_time)
        self.feed_states: dict[str, FeedState] = {
            url: FeedState(
                url=url,
                use_feed_summary=use_feed_summary,
                current_backoff_minutes=float(initial_check_interval),
                next_check_time=now # Explicitly set creation time
            )
            for url in feed_urls
        }
        self.initial_check_interval = float(initial_check_interval) # Store base interval

# --- Core Summarization Logic (extracted from old summarize_new_articles) ---
def process_and_summarize(
    articles_to_process: list,
    processed_ids: set,
    llm: ChatOpenAI,
    model_name: str,
    system_prompt: str,
    output_feed_file: str,
    output_feed_title: str,
    output_feed_link: str,
    output_feed_description: str
) -> set:
    """Processes buffered articles, generates a combined summary, and updates the feed.

    Args:
        articles_to_process (list): List of (entry, content_to_summarize, feed_url) tuples.
        processed_ids (set): The current set of processed IDs.
        llm (ChatOpenAI): The language model instance.
        model_name (str): Name of the LLM model being used.
        system_prompt (str): System prompt for the summary.
        output_feed_file (str): Path to the output RSS file.
        output_feed_title (str): Title for the output RSS feed.
        output_feed_link (str): Link for the output RSS feed.
        output_feed_description (str): Description for the output RSS feed.

    Returns:
        set: The updated set of processed IDs after this run.
    """
    logger.info(f"Starting scheduled summarization process...")

    # Initialize updated_processed_ids early, copying the input set
    updated_processed_ids = processed_ids.copy()

    if not articles_to_process:
        logger.info("No new articles in the buffer to summarize.")
        # Still generate an empty/placeholder feed if requested
        combined_summary = "No new articles to summarize for this period."
        first_article_link = None
        # Skip directly to feed generation and ID saving/return
    else:
        logger.info(f"Preparing summary for {len(articles_to_process)} new articles/entries...")
        articles_processed_this_run = set() # IDs processed in *this* specific summary run
        combined_content_parts = []
        articles_included_count = 0
        first_article_link = None # To use as a fallback link for the summary entry
        included_feeds = set() # Keep track of which feeds contributed

        logger.info("-" * 60)
        logger.info("Articles included in this summary batch:")
        for entry, content_to_summarize, feed_url in articles_to_process:
            title = entry.get('title', 'No Title')
            link = entry.get('link', 'No Link')
            # Use link as ID if guid/id are missing, crucial for processed_ids tracking
            article_id = entry.get('guid') or entry.get('id') or link

            if not first_article_link and link != 'No Link':
                first_article_link = link

            # Add the ID to the set of articles processed in this specific run
            # AND to the main updated_processed_ids set
            if article_id and article_id != 'No Link':
                articles_processed_this_run.add(article_id)
                updated_processed_ids.add(article_id) # <-- FIX: Add to the main set
            else:
                logger.warning(f"  Could not determine a unique ID for article '{title}'. It might be reprocessed.")

            included_feeds.add(feed_url) # Track the source feed

            if content_to_summarize:
                logger.info(f"  - [{feed_url}] {title} ({link})")
                formatted_article = f"Title: {title}\nLink: {link}\n\n{content_to_summarize}\n\n---\n\n"
                combined_content_parts.append(formatted_article)
                articles_included_count += 1
            else:
                logger.info(f"  - Skipping content for '{title}' ({link}) - No content available.")

        combined_summary = None # Initialize to None
        if combined_content_parts:
            logger.info(f"Generating combined summary for {articles_included_count} articles...")
            combined_content = "".join(combined_content_parts)
            full_text_to_summarize = system_prompt + combined_content

            combined_summary = summarize_text_with_langchain(full_text_to_summarize, llm, model_name)

            logger.info("--- Combined Daily Summary ---")
            logger.info(combined_summary)
            logger.info("--- End of Summary ---")
        else:
            logger.info("No content available from buffered articles to generate a combined summary.")
            combined_summary = "No new content available to summarize for this period." # Provide a placeholder

    # --- RSS Feed Generation (Now runs even if articles_to_process was empty) ---
    logger.info(f"Generating RSS feed: {output_feed_file}")
    generate_summary_feed(
        output_feed_file=output_feed_file,
        output_feed_title=output_feed_title,
        output_feed_link=output_feed_link,
        output_feed_description=output_feed_description,
        combined_summary=combined_summary,
        first_article_link=first_article_link
    )

    # Persist the processed IDs
    # If articles_to_process was empty, updated_processed_ids is still the original set
    # If it wasn't empty, updated_processed_ids now contains the newly processed IDs
    save_processed_ids(updated_processed_ids)

    logger.info("-" * 60)
    # Adjust logging based on whether articles were actually processed
    if articles_to_process:
        logger.info(f"Finished scheduled summarization. Processed IDs in this batch: {len(articles_processed_this_run)}.")
        logger.info(f"Total unique processed IDs known: {len(updated_processed_ids)}.")
        if combined_content_parts:
            logger.info(f"Included content from {articles_included_count} articles from {len(included_feeds)} feeds in the summary generation.")
        else:
            logger.info("No new articles with content were available to generate a summary.")
    else:
        logger.info("Finished scheduled summarization. No new articles were processed.")
        logger.info(f"Total unique processed IDs known: {len(updated_processed_ids)}.")

    return updated_processed_ids # Return the potentially updated set

# --- Scheduling Wrappers ---
def scheduled_check_feed_job(app_state: AppState, feed_state: FeedState):
    """Wrapper function for scheduling feed checks for a *single* feed. Updates AppState."""
    feed_url = feed_state.url
    use_feed_summary = feed_state.use_feed_summary

    # Set is_checking flag under lock
    with app_state.lock:
        if feed_state.is_checking:
            logger.warning(f"Check for {feed_url} already in progress. Skipping.")
            return
        feed_state.is_checking = True

    logger.debug(f"Running scheduled check for {feed_url}...")
    now = datetime.datetime.now()
    success = False
    new_entries = []
    updated_ids = set()

    try:
        # Pass the current processed IDs from state
        with app_state.lock:
            current_ids = app_state.processed_ids.copy()

        new_entries_tuples, updated_ids = check_feed(feed_url, use_feed_summary, current_ids)
        # Add feed_url to each new entry tuple for tracking in the buffer
        new_entries = [(entry, content, feed_url) for entry, content in new_entries_tuples]
        success = True

    except Exception as e:
        logger.error(f"Error during scheduled feed check for {feed_url}: {e}", exc_info=False) # Don't need full trace usually
        success = False
    finally:
        # Update state under lock regardless of success/failure
        with app_state.lock:
            feed_state.last_check_time = now
            feed_state.is_checking = False # Reset checking flag

            if success:
                app_state.articles_buffer.extend(new_entries)
                app_state.processed_ids = updated_ids # Update with the set returned by check_feed
                feed_state.failure_count = 0 # Reset failures on success
                feed_state.current_backoff_minutes = app_state.initial_check_interval # Reset backoff on success
                logger.info(f"Successfully checked {feed_url}. Added {len(new_entries)} new items. Next check in ~{feed_state.current_backoff_minutes:.1f} min.")
            else:
                # Failure: Increase failure count and calculate next backoff
                feed_state.failure_count += 1
                backoff = feed_state.current_backoff_minutes * (BACKOFF_FACTOR ** feed_state.failure_count)
                # Apply jitter: +/- JITTER_FACTOR %
                jitter = random.uniform(-JITTER_FACTOR, JITTER_FACTOR)
                backoff_with_jitter = backoff * (1 + jitter)
                # Cap backoff at MAX_BACKOFF_MINUTES
                feed_state.current_backoff_minutes = min(backoff_with_jitter, MAX_BACKOFF_MINUTES)
                logger.warning(f"Failed check for {feed_url} (Attempt {feed_state.failure_count}). Backing off. Next check in ~{feed_state.current_backoff_minutes:.1f} min.")

            # Schedule next check time
            feed_state.next_check_time = now + datetime.timedelta(minutes=feed_state.current_backoff_minutes)
            logger.debug(f"Feed {feed_url}: Next check scheduled for {feed_state.next_check_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Log buffer/ID info outside the lock if needed, using potentially slightly stale data
    logger.debug(f"Feed check complete for {feed_url}. Buffer size: {len(app_state.articles_buffer)}, Processed IDs: {len(app_state.processed_ids)}")

def scheduled_summarize_job(app_state: AppState, llm: ChatOpenAI, model_name: str, feed_args: dict, system_prompt: str):
    """Wrapper function for scheduling summarization. Updates AppState."""
    logger.debug("Running scheduled summarization job...")

    # Get buffer and current IDs, clear buffer under lock
    with app_state.lock:
        buffer_copy = list(app_state.articles_buffer)
        app_state.articles_buffer.clear()
        current_ids = app_state.processed_ids.copy()

    if not buffer_copy:
        logger.info("Scheduled summary run: No articles in buffer to process.")
        return # Nothing to do

    try:
        # Call the main processing function
        updated_ids = process_and_summarize(
            articles_to_process=buffer_copy,
            processed_ids=current_ids,
            llm=llm,
            model_name=model_name,
            system_prompt=system_prompt,
            **feed_args # Pass feed generation arguments directly
        )

        # Update the master processed_ids list in state under lock
        with app_state.lock:
            app_state.processed_ids = updated_ids
            logger.debug(f"Summarization complete. Processed IDs: {len(app_state.processed_ids)}")

    except Exception as e:
        logger.error(f"Error during scheduled summarization processing: {e}", exc_info=True)
        # Decide if buffer_copy should be restored on failure
        # For now, articles are considered 'lost' for this cycle if summarization fails critically.
        # They were already marked processed by check_feed, preventing reprocessing.

# --- Main Scheduling Loop --- (Can be run in a thread)
def run_scheduler(app_state: AppState, check_interval: int, summary_time: str, llm: ChatOpenAI, model_name: str, feed_args: dict, system_prompt: str):
    """Sets up and runs the schedule loop."""

    # --- Schedule the jobs ---
    # The feed check is no longer scheduled with schedule library directly.
    # We manage the timing within the loop based on feed_state.next_check_time.
    logger.info(f"Feed checker started. Initial check interval: {app_state.initial_check_interval} minutes (will adjust with backoff).")

    # Schedule the daily summary
    schedule.every().day.at(summary_time).do(
        scheduled_summarize_job,
        app_state=app_state,
        llm=llm,
        model_name=model_name,
        feed_args=feed_args,
        system_prompt=system_prompt
    )
    logger.info(f"Scheduled daily summary at {summary_time}.")

    logger.info("Scheduler started. Waiting for pending jobs...")

    # --- Run Loop ---
    while True:
        now = datetime.datetime.now()
        feeds_to_check = []

        # --- Check which feeds are due ---
        with app_state.lock:
            for feed_url, feed_state in app_state.feed_states.items():
                if not feed_state.is_checking and now >= feed_state.next_check_time:
                    feeds_to_check.append(feed_state) # Add the state object

        # --- Trigger checks for due feeds (outside the lock) ---
        if feeds_to_check:
            logger.debug(f"Found {len(feeds_to_check)} feeds due for checking.")
            for feed_state in feeds_to_check:
                 # Run checks in separate threads? For now, run sequentially.
                 # If running in threads, need to manage thread pool etc.
                 try:
                     scheduled_check_feed_job(app_state, feed_state)
                 except Exception as e:
                      # Catch errors here too, although the job itself handles internal errors
                      logger.error(f"Unexpected error launching check for {feed_state.url}: {e}", exc_info=True)
                      # We might want to reset the is_checking flag or handle this failure state more robustly
                      # For now, the finally block in the job should reset is_checking.

        # --- Run scheduled summary job (using schedule library) ---
        try:
            schedule.run_pending()
        except Exception as e:
            logger.error(f"Error running schedule.run_pending(): {e}", exc_info=True)

        # --- Sleep ---
        try:
            # Sleep for a short duration before checking again
            time.sleep(10) # Check feed states every 10 seconds
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in scheduler loop. Stopping scheduler...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in scheduler main loop: {e}", exc_info=True)
            time.sleep(60) # Sleep longer if there's a loop error

    logger.info("Scheduler finished.") 