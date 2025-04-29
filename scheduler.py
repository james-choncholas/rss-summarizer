import schedule
import time
import threading # For potential state locking if needed

# Import necessary components from other modules
from config import logger
from feed import check_feed
from summarization import summarize_text_with_langchain
from output_feed import generate_summary_feed
from utils import save_processed_ids # Need this for final save in summarization
from langchain_openai import ChatOpenAI # Type hint for llm

# --- State Management ---
# Use a class or dictionary to manage state passed between scheduled jobs
class AppState:
    def __init__(self, processed_ids):
        self.processed_ids = processed_ids
        self.articles_buffer = [] # List to store (entry, content_to_summarize)
        self.lock = threading.Lock() # Lock for thread-safe access to buffer/ids

# --- Core Summarization Logic (extracted from old summarize_new_articles) ---
def process_and_summarize(
    articles_to_process: list,
    processed_ids: set,
    llm: ChatOpenAI,
    model_name: str,
    output_feed_file: str,
    output_feed_title: str,
    output_feed_link: str,
    output_feed_description: str
) -> set:
    """Processes buffered articles, generates a combined summary, and updates the feed.

    Args:
        articles_to_process (list): List of (entry, content_to_summarize) tuples.
        processed_ids (set): The current set of processed IDs.
        llm (ChatOpenAI): The language model instance.
        model_name (str): Name of the LLM model being used.
        output_feed_file (str): Path to the output RSS file.
        output_feed_title (str): Title for the output RSS feed.
        output_feed_link (str): Link for the output RSS feed.
        output_feed_description (str): Description for the output RSS feed.

    Returns:
        set: The updated set of processed IDs after this run.
    """
    logger.info(f"Starting scheduled summarization process...")

    if not articles_to_process:
        logger.info("No new articles in the buffer to summarize.")
        return processed_ids # Return unchanged set

    logger.info(f"Preparing summary for {len(articles_to_process)} new articles/entries...")
    articles_processed_this_run = set() # IDs processed in *this* specific summary run
    combined_content_parts = []
    articles_included_count = 0
    first_article_link = None # To use as a fallback link for the summary entry
    updated_processed_ids = processed_ids.copy() # Work on a copy for this run

    logger.info("-" * 60)
    logger.info("Articles included in this summary batch:")
    for entry, content_to_summarize in articles_to_process:
        title = entry.get('title', 'No Title')
        link = entry.get('link', 'No Link')
        # Use link as ID if guid/id are missing, crucial for processed_ids tracking
        article_id = entry.get('guid') or entry.get('id') or link

        if not first_article_link and link != 'No Link':
            first_article_link = link

        # Add the ID to the set of articles processed in this specific run
        # We already added it to updated_processed_ids in check_feed, but we track
        # it here again to know which IDs correspond to *this batch*
        if article_id and article_id != 'No Link':
            articles_processed_this_run.add(article_id)
        else:
            logger.warning(f"  Could not determine a unique ID for article '{title}'. It might be reprocessed.")

        if content_to_summarize:
            logger.info(f"  - {title} ({link})")
            formatted_article = f"Title: {title}\nLink: {link}\n\n{content_to_summarize}\n\n---\n\n"
            combined_content_parts.append(formatted_article)
            articles_included_count += 1
        else:
            logger.info(f"  - Skipping content for '{title}' ({link}) - No content available.")

    combined_summary = None # Initialize to None
    if combined_content_parts:
        logger.info(f"Generating combined summary for {articles_included_count} articles...")
        combined_content = "".join(combined_content_parts)
        summary_prompt_prefix = f"Provide a concise combined summary of the following {articles_included_count} articles:\n\n"
        full_text_to_summarize = summary_prompt_prefix + combined_content

        combined_summary = summarize_text_with_langchain(full_text_to_summarize, llm, model_name)

        logger.info("--- Combined Daily Summary ---")
        logger.info(combined_summary)
        logger.info("--- End of Summary ---")
    else:
        logger.info("No content available from buffered articles to generate a combined summary.")
        combined_summary = "No new content available to summarize for this period." # Provide a placeholder

    # --- RSS Feed Generation ---
    # Call the dedicated feed generation function
    generate_summary_feed(
        output_feed_file=output_feed_file,
        output_feed_title=output_feed_title,
        output_feed_link=output_feed_link,
        output_feed_description=output_feed_description,
        combined_summary=combined_summary,
        first_article_link=first_article_link
    )

    # Persist the processed IDs (this was done in check_feed, but saving again ensures consistency
    # if summarize runs less frequently or if check_feed saving failed)
    # Note: save_processed_ids overwrites the file with the current set.
    save_processed_ids(updated_processed_ids)

    logger.info("-" * 60)
    logger.info(f"Finished scheduled summarization. Processed IDs in this batch: {len(articles_processed_this_run)}.")
    logger.info(f"Total unique processed IDs known: {len(updated_processed_ids)}.")
    if combined_content_parts:
         logger.info(f"Included content from {articles_included_count} articles in the summary generation.")

    return updated_processed_ids # Return the final updated set

# --- Scheduling Wrappers --- a
def scheduled_check_feed_job(app_state: AppState, feed_url: str, use_feed_summary: bool):
    """Wrapper function for scheduling feed checks. Updates AppState."""
    logger.debug(f"Running scheduled check for {feed_url}...")
    try:
        # Pass the current processed IDs from state
        with app_state.lock:
            current_ids = app_state.processed_ids.copy()

        new_entries, updated_ids = check_feed(feed_url, use_feed_summary, current_ids)

        # Update state under lock
        with app_state.lock:
            app_state.articles_buffer.extend(new_entries)
            app_state.processed_ids = updated_ids # Update with the set returned by check_feed
            logger.debug(f"Feed check complete. Buffer size: {len(app_state.articles_buffer)}, Processed IDs: {len(app_state.processed_ids)}")

    except Exception as e:
        logger.error(f"Error during scheduled feed check: {e}", exc_info=True)

def scheduled_summarize_job(app_state: AppState, llm: ChatOpenAI, model_name: str, feed_args: dict):
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
def run_scheduler(app_state: AppState, check_interval: int, summary_time: str, feed_url: str, use_feed_summary: bool, llm: ChatOpenAI, model_name: str, feed_args: dict):
    """Sets up and runs the schedule loop."""

    # --- Schedule the jobs --- 
    # Schedule the feed check
    schedule.every(check_interval).minutes.do(
        scheduled_check_feed_job,
        app_state=app_state,
        feed_url=feed_url,
        use_feed_summary=use_feed_summary
    )
    logger.info(f"Scheduled feed check every {check_interval} minutes.")

    # Schedule the daily summary
    schedule.every().day.at(summary_time).do(
        scheduled_summarize_job,
        app_state=app_state,
        llm=llm,
        model_name=model_name,
        feed_args=feed_args
    )
    logger.info(f"Scheduled daily summary at {summary_time}.")

    logger.info("Scheduler started. Waiting for pending jobs...")

    # --- Run Loop ---
    while True:
        try:
            schedule.run_pending()
            time.sleep(60) # Check every minute
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in scheduler loop. Stopping scheduler...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in scheduler loop: {e}", exc_info=True)
            # Optionally add a delay before retrying to prevent fast looping on errors
            time.sleep(60)

    logger.info("Scheduler finished.") 