import argparse
import datetime
import feedgen
import http.server
import json
import logging # Added
import os
import schedule
import socketserver
import threading
import time
import tiktoken # Add this import for token counting

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from feedgen.feed import FeedGenerator
from langchain.prompts import ChatPromptTemplate
# from langchain.chains.summarize import load_summarize_chain # Alternative for long docs
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# --- Configuration & Constants ---
load_dotenv() # Load environment variables first

# --- Environment & API ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in the .env file.")

# --- Behavior ---
REQUEST_DELAY_SECONDS = 1  # Delay between fetching full articles
REQUEST_TIMEOUT_SECONDS = 15 # Timeout for fetching article content
USER_AGENT = "RSSSummarizerBot/1.0 (+https://github.com/your-repo/rss-summarizer)" # User agent for requests
CHECK_INTERVAL_MINUTES = 30 # How often to check the feed (in minutes)
SUMMARY_TIME = "17:00" # Time to run the daily summary (HH:MM)
MAX_TOKENS = 4096 # Max tokens for the LLM input (including prompt)

# --- Files & Paths ---
PROCESSED_IDS_FILE = "processed_article_ids.json" # File to store processed article IDs

# --- Output Feed Defaults ---
DEFAULT_OUTPUT_FEED_FILE = "summary_feed.xml"
DEFAULT_OUTPUT_FEED_TITLE = "Daily RSS Summary"
DEFAULT_OUTPUT_FEED_LINK = "http://localhost:8000/summary_feed.xml" # Default includes port now
DEFAULT_OUTPUT_FEED_DESC = "A daily summary of articles from monitored RSS feeds."

# --- Server ---
DEFAULT_SERVER_PORT = 8000

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Global State (To be refactored/minimized later) ---
processed_ids = set()
new_articles_buffer = [] # Stores (entry, content_to_summarize) tuples

# --- Helper Functions ---

def load_processed_ids(filename=PROCESSED_IDS_FILE):
    """Loads processed article IDs from a JSON file."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return set(json.load(f))
        else:
            return set()
    except (IOError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load processed IDs from {filename}: {e}. Starting fresh.")
        return set()

def save_processed_ids(ids_set, filename=PROCESSED_IDS_FILE):
    """Saves processed article IDs to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(list(ids_set), f) # Convert set to list for JSON serialization
    except IOError as e:
        logger.error(f"Error saving processed IDs to {filename}: {e}")

def fetch_rss_feed(feed_url):
    """Fetches and parses the RSS feed."""
    logger.info(f"Fetching RSS feed: {feed_url}")
    try:
        feed = feedparser.parse(feed_url)
        if feed.bozo:
            logger.warning(f"Feed may be malformed. Bozo reason: {feed.bozo_exception}")
        if not feed.entries:
            logger.warning("No entries found in the feed.")
        return feed.entries
    except Exception as e:
        logger.error(f"Error fetching or parsing feed {feed_url}: {e}")
        return []

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count the number of tokens in a text string."""
    if not text: return 0 # Handle empty text
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

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
        return clean_text(content) # Clean the final joined text

    except requests.exceptions.RequestException as e:
        logger.error(f"  Error fetching article {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"  Error parsing article {url}: {e}")
        return None

def summarize_text_with_langchain(text_to_summarize, llm):
    """Summarizes the given text using LangChain."""
    if not text_to_summarize:
         logger.info("  Skipping summary for empty text.")
         return "Content unavailable to summarize."

    logger.info("  Summarizing text...")

    # Check token count BEFORE creating prompt/chain
    # Estimate max tokens for the prompt itself
    prompt_overhead_estimate = 50
    available_tokens_for_text = MAX_TOKENS - prompt_overhead_estimate
    text_tokens = count_tokens(text_to_summarize)

    if text_tokens > available_tokens_for_text:
        logger.warning(f"  Warning: Text ({text_tokens} tokens) too long for model's {MAX_TOKENS} limit (estimated). Truncating...")
        # Simple truncation based on estimated character ratio
        # A more robust method might involve token-aware truncation
        estimated_chars_per_token = len(text_to_summarize) / text_tokens
        max_chars = int(available_tokens_for_text * estimated_chars_per_token * 0.95) # 5% buffer
        text_to_summarize = text_to_summarize[:max_chars]
        logger.warning(f"  Truncated text to ~{count_tokens(text_to_summarize)} tokens.")


    # Use a slightly more robust prompt
    prompt_template = """Please provide a concise summary of the following text. Focus on the key information and main points.

    TEXT:
    "{text}"

    CONCISE SUMMARY:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | llm | StrOutputParser()

    try:
        summary = chain.invoke({"text": text_to_summarize})
        logger.info("  Summary generated.")
        # Basic post-processing: remove potential leading/trailing whitespace
        return summary.strip()
    except Exception as e:
        logger.error(f"  Error during summarization: {e}")
        # Consider more specific error handling (e.g., context length exceeded)
        # You might want to check for specific error types from OpenAI/Langchain
        if "context_length_exceeded" in str(e).lower():
             return "Error: The combined text was too long to summarize, even after truncation."
        return "Error generating summary."

# --- Core Logic ---

def check_feed(feed_url, use_feed_summary, processed_ids):
    """Checks the feed for new articles and adds them to a buffer."""
    logger.info(f"Checking feed: {feed_url}")
    entries = fetch_rss_feed(feed_url)
    new_entries_found = [] # Local buffer for this check run
    updated_processed_ids = processed_ids.copy() # Work on a copy

    for entry in entries:
        article_id = entry.get('guid') or entry.get('id') or entry.get('link')
        if not article_id:
            logger.warning(f"  Skipping entry with no guid, id, or link: {entry.get('title', 'No Title')}")
            continue

        if article_id not in updated_processed_ids:
            title = entry.get('title', 'No Title')
            link = entry.get('link')
            logger.info(f"  Found new article: {title}")
            updated_processed_ids.add(article_id)

            content_to_summarize = None
            if use_feed_summary:
                logger.info("    Using summary/description from feed entry.")
                content_to_summarize = entry.get('summary') or entry.get('description')
                if not content_to_summarize:
                     logger.warning("    No summary/description found in feed entry.")
            elif link:
                 # Fetch content immediately if not using feed summary
                 content_to_summarize = fetch_article_content(link)
                 time.sleep(REQUEST_DELAY_SECONDS) # Delay even if fetch fails
                 if not content_to_summarize:
                      logger.warning(f"    Failed to fetch content for '{title}'.")
            else:
                 logger.warning(f"    No link found for '{title}' and not using feed summary.")


            # Buffer the entry and the content (even if None)
            new_entries_found.append((entry, content_to_summarize))
            # else: # No need for this else block now
            # logger.info(f"  Skipping already processed article: {entry.get('title', 'No Title')}")
            # pass # Keep pass here to make diff smaller

    logger.info(f"Finished checking feed. Found {len(new_entries_found)} new articles.")
    if new_entries_found:
         save_processed_ids(updated_processed_ids)

    # Return the new entries found and the updated set of processed IDs
    return new_entries_found, updated_processed_ids


def summarize_new_articles(llm, output_feed_file, output_feed_title, output_feed_link, output_feed_description, articles_to_process, processed_ids):
    """Combines and summarizes articles, then adds the summary as a new entry to an RSS feed."""
    logger.info(f"Starting scheduled summarization and feed generation...")

    if not articles_to_process:
        logger.info("No new articles in the buffer to summarize.")
        return processed_ids # Return the unchanged processed_ids

    logger.info(f"Preparing summary for {len(articles_to_process)} new articles/entries...")
    articles_processed_this_run = set()
    combined_content_parts = []
    articles_included_count = 0
    first_article_link = None # To use as a fallback link for the summary entry
    updated_processed_ids = processed_ids.copy() # Work on a copy

    # No need to clear a global buffer now

    logger.info("-" * 60)
    logger.info("Articles included in this summary batch:")
    for entry, content_to_summarize in articles_to_process: # Use the passed list directly
        title = entry.get('title', 'No Title')
        link = entry.get('link', 'No Link')
        # Use link as ID if guid/id are missing, crucial for processed_ids tracking
        article_id = entry.get('guid') or entry.get('id') or link

        if not first_article_link and link != 'No Link':
            first_article_link = link

        # Track ID for marking as processed later, even if content is missing
        if article_id and article_id != 'No Link': # Ensure we have a valid ID
            articles_processed_this_run.add(article_id)
        else:
            # If no ID, we can't reliably mark it processed. Log warning.
            logger.warning(f"  Could not determine a unique ID for article '{title}'. It may be reprocessed if it appears again.")

        if content_to_summarize:
            logger.info(f"  - {title} ({link})")
            # Format for combining. Add more context if needed.
            formatted_article = f"Title: {title}\nLink: {link}\n\n{content_to_summarize}\n\n---\n\n"
            combined_content_parts.append(formatted_article)
            articles_included_count += 1
        else:
            # Still log the article title even if skipped for content
            logger.info(f"  - Skipping content for '{title}' ({link}) - No content was fetched or found in feed.")

    combined_summary = "No summary generated (no content available)." # Default
    if combined_content_parts:
        logger.info(f"Generating combined summary for {articles_included_count} articles...")
        combined_content = "".join(combined_content_parts)
        # Simplified prompt prefix
        summary_prompt_prefix = f"Provide a concise combined summary of the following {articles_included_count} articles:\n\n"
        full_text_to_summarize = summary_prompt_prefix + combined_content

        # Generate the summary
        combined_summary = summarize_text_with_langchain(full_text_to_summarize, llm)

        # Print summary to console as well for logging/debugging
        logger.info("--- Combined Daily Summary ---")
        logger.info(combined_summary)
        logger.info("--- End of Summary ---")
    else:
        logger.info("No content available from buffered articles to generate a combined summary.")
        # Decide if we should still create a feed entry (e.g., "No new content summarized today")
        # For now, we'll skip creating an entry if there was absolutely no content.
        logger.info("Skipping RSS feed update as there was no content to summarize.")
        # Still need to mark articles as processed
        updated_processed_ids.update(articles_processed_this_run)
        save_processed_ids(updated_processed_ids)
        logger.info(f"Finished scheduled run. Attempted to process {len(articles_processed_this_run)} unique article IDs. No feed generated.")
        return updated_processed_ids # Return updated IDs

    # --- RSS Feed Generation ---
    logger.info(f"\nGenerating RSS feed entry and updating {output_feed_file}...")
    fg = FeedGenerator()

    # Initialize feed metadata (do this every time to potentially update it)
    fg.title(output_feed_title)
    fg.link(href=output_feed_link, rel='alternate')
    fg.description(output_feed_description)
    fg.language('en') # Or make configurable

    # Try loading existing feed *after* setting metadata
    # This ensures metadata is present even if loading fails or file is new
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
                    old_fe.published(datetime.datetime(*entry.published_parsed[:6], tzinfo=datetime.timezone.utc)) # Assume UTC if parsed
                if entry.get('updated_parsed'):
                     old_fe.updated(datetime.datetime(*entry.updated_parsed[:6], tzinfo=datetime.timezone.utc)) # Assume UTC if parsed
                # Handle content (feedparser puts it in a list)
                if entry.get('content'):
                     old_fe.content(entry.content[0].value, type=entry.content[0].type)
                # Copy other potential fields if needed (e.g., author)

            logger.info(f"  Loaded {len(existing_feed.entries)} existing entries from feed file.")
        except Exception as e:
            logger.warning(f"  Warning: Could not load or parse existing feed file {output_feed_file}: {e}. Starting with a new feed.", exc_info=True)
            # Proceed with the initialized fg object (new feed)
    else:
         logger.info("  No existing feed file found. Creating a new feed.")


    # Create a new feed entry for this summary
    fe = fg.add_entry(order='prepend') # Add new entries to the top
    summary_title = f"Summary for {datetime.date.today().strftime('%Y-%m-%d')}"
    fe.title(summary_title)
    # Use a unique ID based on the date
    # Consider adding time if summaries could happen more than once a day
    entry_id = f"urn:summary:{datetime.date.today().strftime('%Y%m%d')}:{output_feed_title}"
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
    fe.content(combined_summary or "Summary could not be generated.", type='text')


    # Save the updated feed
    try:
        fg.rss_file(output_feed_file, pretty=True) # Save as RSS 2.0
        logger.info(f"  Successfully updated RSS feed: {output_feed_file}")
    except Exception as e:
        logger.error(f"  Error saving RSS feed to {output_feed_file}: {e}")

    # Update the main processed set and save
    updated_processed_ids.update(articles_processed_this_run)
    save_processed_ids(updated_processed_ids)

    logger.info("-" * 60)
    logger.info(f"Finished scheduled summarization. Attempted to process {len(articles_processed_this_run)} unique article IDs.")
    if combined_content_parts:
         logger.info(f"Successfully included content from {articles_included_count} articles in the combined summary generation.")

    return updated_processed_ids # Return the final updated set


# --- HTTP Server ---

def run_http_server(port, directory="."):
    """Runs a simple HTTP server in the current directory."""
    Handler = http.server.SimpleHTTPRequestHandler
    # Change directory before starting the server if needed
    # os.chdir(directory) # Not strictly needed if running from the script's dir

    # Use socketserver.TCPServer for better handling, especially thread management
    # Allow address reuse to prevent errors on quick restarts
    socketserver.TCPServer.allow_reuse_address = True
    # Determine the output feed filename dynamically - needed for the log message
    # We can't easily get args here, so let's use the default or assume it's passed
    # For simplicity in this function, let's just print a generic message or get the filename globally?
    # Best practice: Pass the specific filename if needed, but SimpleHTTPRequestHandler serves all files.

    # Let's refine the print message slightly
    logger.info(f"Starting HTTP server on port {port} to serve files from the current directory.")
    logger.info(f"Access the generated feed typically at http://localhost:{port}/{DEFAULT_OUTPUT_FEED_FILE}")

    with socketserver.TCPServer(("", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            # This might not be caught here if the main thread catches it first
            logger.info("HTTP server received KeyboardInterrupt (likely from main thread). Shutting down...")
        except Exception as e:
            # Log unexpected errors in the server thread
            logger.error(f"HTTP server error: {e}", exc_info=True)
        finally:
            # Ensure cleanup happens
            httpd.server_close() # Close the server socket
            logger.info(f"HTTP server on port {port} stopped.")

# --- Scheduling Wrappers ---

# Global buffer to hold articles between checks and summarization
# We keep this global for now as schedule doesn't easily pass state between jobs
# A class-based approach would encapsulate this better.
articles_buffer = []
current_processed_ids = set()

def scheduled_check_feed(feed_url, use_feed_summary):
    """Wrapper function for scheduling feed checks."""
    global articles_buffer, current_processed_ids
    logger.debug("Running scheduled feed check...")
    try:
        new_entries, updated_ids = check_feed(feed_url, use_feed_summary, current_processed_ids)
        articles_buffer.extend(new_entries) # Add new entries to the global buffer
        current_processed_ids = updated_ids # Update the global set
        logger.debug(f"Feed check complete. Buffer size: {len(articles_buffer)}, Processed IDs: {len(current_processed_ids)}")
    except Exception as e:
        logger.error(f"Error during scheduled feed check: {e}", exc_info=True)


def scheduled_summarize(llm, output_feed_file, output_feed_title, output_feed_link, output_feed_description):
    """Wrapper function for scheduling summarization."""
    global articles_buffer, current_processed_ids
    logger.debug("Running scheduled summarization...")
    # Pass a copy of the buffer and clear the global one *before* processing
    buffer_copy = list(articles_buffer)
    articles_buffer.clear()
    try:
        updated_ids = summarize_new_articles(llm, output_feed_file, output_feed_title, output_feed_link, output_feed_description, buffer_copy, current_processed_ids)
        current_processed_ids = updated_ids # Update the global set with the result
        logger.debug(f"Summarization complete. Processed IDs: {len(current_processed_ids)}")
    except Exception as e:
        logger.error(f"Error during scheduled summarization: {e}", exc_info=True)
        # Decide if we need to put buffer_copy back into articles_buffer on failure
        # For now, we assume summarize_new_articles handles internal errors and updates processed_ids partially if needed.

# --- Main Execution ---

def main():
    # global processed_ids # No longer needed
    global current_processed_ids # Still needed for the scheduler wrappers

    parser = argparse.ArgumentParser(description="Monitor an RSS feed, summarize new articles daily, publish to a new RSS feed, and serve it.")
    parser.add_argument("feed_url", help="The URL of the RSS feed to monitor.")
    parser.add_argument("--use_feed_summary", action="store_true", help="Use summary/description from feed entry directly instead of fetching full article content.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model name to use (default: gpt-3.5-turbo).")
    parser.add_argument("--temperature", type=float, default=0.3, help="LLM temperature (creativity). Lower is more deterministic (default: 0.3).")
    parser.add_argument("--check_interval", type=int, default=CHECK_INTERVAL_MINUTES, help=f"How often to check the feed in minutes (default: {CHECK_INTERVAL_MINUTES}).")
    parser.add_argument("--summary_time", type=str, default=SUMMARY_TIME, help=f"Time to run the daily summary in HH:MM format (24-hour clock) (default: {SUMMARY_TIME}).")
    parser.add_argument("--run_once", action="store_true", help="Run the check and summary once immediately, then exit (for testing).")
    # New arguments for output feed
    parser.add_argument("--output_feed_file", type=str, default=DEFAULT_OUTPUT_FEED_FILE, help=f"Filename for the generated summary RSS feed (default: {DEFAULT_OUTPUT_FEED_FILE}).")
    parser.add_argument("--output_feed_title", type=str, default=DEFAULT_OUTPUT_FEED_TITLE, help=f"Title for the generated RSS feed (default: {DEFAULT_OUTPUT_FEED_TITLE}).")
    parser.add_argument("--output_feed_link", type=str, default=DEFAULT_OUTPUT_FEED_LINK, help=f"Link for the generated RSS feed (default: {DEFAULT_OUTPUT_FEED_LINK}).")
    parser.add_argument("--output_feed_description", type=str, default=DEFAULT_OUTPUT_FEED_DESC, help=f"Description for the generated RSS feed (default: {DEFAULT_OUTPUT_FEED_DESC}).")
    # New argument for server port
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT, help=f"Port to serve the RSS feed on (default: {DEFAULT_SERVER_PORT}).")


    args = parser.parse_args()

    # Validate summary_time format
    try:
        datetime.datetime.strptime(args.summary_time, '%H:%M')
    except ValueError:
        logger.error(f"Invalid summary_time format '{args.summary_time}'. Please use HH:MM (24-hour clock).")
        exit(1)


    current_processed_ids = load_processed_ids() # Load initial IDs into the global set
    logger.info(f"Loaded {len(current_processed_ids)} processed article IDs from {PROCESSED_IDS_FILE}.")

    llm = ChatOpenAI(
        model_name=args.model,
        temperature=args.temperature,
        openai_api_key=OPENAI_API_KEY # Use the constant loaded earlier
    )

    logger.info("--- RSS Summarizer Bot Started ---")
    logger.info(f"Monitoring Feed: {args.feed_url}")
    logger.info(f"Checking every: {args.check_interval} minutes")
    logger.info(f"Daily Summary Time: {args.summary_time}")
    logger.info(f"Using feed summary directly: {args.use_feed_summary}")
    logger.info(f"LLM Model: {args.model}")
    logger.info(f"Output Feed File: {args.output_feed_file}")
    logger.info(f"Output Feed Title: {args.output_feed_title}")
    # Update default link based on actual port
    default_link = f"http://localhost:{args.port}/{args.output_feed_file}"
    output_link = args.output_feed_link if args.output_feed_link != DEFAULT_OUTPUT_FEED_LINK else default_link
    logger.info(f"Output Feed Link: {output_link}") # Use potentially updated link
    logger.info(f"Output Feed Description: {args.output_feed_description}")
    logger.info(f"Serving feed on port: {args.port}") # Added log for port
    logger.info("------------------------------------")

    # --- Start HTTP Server in a separate thread ---
    server_thread = threading.Thread(
        target=run_http_server,
        args=(args.port,),
        daemon=True # Set as daemon so it exits when the main thread exits cleanly
    )
    server_thread.start()
    # Give the server a moment to potentially print its startup messages
    time.sleep(0.5)

    # Perform an initial check immediately on startup
    logger.info("Performing initial feed check...")
    # Initial check now uses the scheduled wrapper to update global state correctly
    scheduled_check_feed(args.feed_url, args.use_feed_summary)


    # Define summary function arguments dictionary
    summary_kwargs = {
        'llm': llm,
        'output_feed_file': args.output_feed_file,
        'output_feed_title': args.output_feed_title,
        'output_feed_link': output_link, # Pass the potentially updated link
        'output_feed_description': args.output_feed_description
    }
    # No longer need bound_summarize_func, will call the wrapper

    if args.run_once:
        logger.info("Running summary immediately due to --run_once flag...")
        # Call the scheduled wrapper which handles global state
        scheduled_summarize(**summary_kwargs)
        logger.info(f"Run once complete. Feed generated and served at {output_link}")
        logger.info("Press Ctrl+C to stop serving.")
        try:
            # Keep main thread alive only to keep server thread running
            while server_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
             logger.info("Exiting after run_once.")
        return

    # --- Setup Scheduling ---
    # Schedule the feed check using the wrapper
    schedule.every(args.check_interval).minutes.do(
        scheduled_check_feed,
        feed_url=args.feed_url,
        use_feed_summary=args.use_feed_summary
    )

    # Schedule the daily summary using the wrapper and kwargs
    schedule.every().day.at(args.summary_time).do(scheduled_summarize, **summary_kwargs)


    logger.info(f"Scheduler started. Waiting for the next check (every {args.check_interval} mins) or summary time ({args.summary_time})...")
    # Log the exact feed URL based on arguments
    logger.info(f"Feed will be available at {output_link}")

    # Main loop - now also implicitly keeps the server thread alive
    try:
        while True:
            schedule.run_pending()
            # Check if the server thread is still alive
            if not server_thread.is_alive():
                 logger.error("HTTP server thread has stopped unexpectedly. Attempting restart...")
                 # Attempt to restart the server thread
                 try:
                      server_thread = threading.Thread(target=run_http_server, args=(args.port,), daemon=True)
                      server_thread.start()
                      time.sleep(0.5) # Give it a moment
                      if not server_thread.is_alive(): # Check again
                           raise RuntimeError("Failed to restart server thread.")
                      logger.info("HTTP server thread restarted successfully.")
                 except Exception as e:
                      logger.critical(f"CRITICAL ERROR: Failed to restart HTTP server thread: {e}. Exiting.", exc_info=True)
                      # Depending on requirements, might want to exit or just log heavily
                      break # Exit the main loop

            time.sleep(60) # Check schedule every minute
    except KeyboardInterrupt:
         logger.info("Ctrl+C received. Shutting down scheduler...")
         # Server thread is daemon, should exit. Explicit shutdown is handled in run_http_server.
    except Exception as e:
        # Catch unexpected errors in the main loop
        logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
    finally:
         # Any other cleanup if needed
         logger.info("Main process exiting.")

if __name__ == "__main__":
    main()