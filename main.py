import os
import feedparser
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
# from langchain.chains.summarize import load_summarize_chain # Alternative for long docs
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import argparse
import time  # To add delays between requests
import tiktoken  # Add this import for token counting
import schedule # Added
import datetime # Added
import json # Added for saving/loading processed IDs
import feedgen # Added for RSS feed generation
from feedgen.feed import FeedGenerator # Added for RSS feed generation
import datetime # Ensure datetime is available for feed generation
import os # Ensure os is available for path checking
import threading # Added for HTTP server
import http.server # Added for HTTP server
import socketserver # Added for HTTP server

# --- Configuration ---
# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

# Check if API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in the .env file.")

# --- Constants ---
REQUEST_DELAY_SECONDS = 1  # Delay between fetching full articles to be polite
REQUEST_TIMEOUT_SECONDS = 15 # Timeout for fetching article content
USER_AGENT = "RSSSummarizerBot/1.0 (+https://github.com/your-repo/rss-summarizer)" # Be a good citizen
PROCESSED_IDS_FILE = "processed_article_ids.json" # File to store processed article IDs
CHECK_INTERVAL_MINUTES = 30 # How often to check the feed (in minutes)
SUMMARY_TIME = "17:00" # Time to run the daily summary
MAX_TOKENS = 4096 # Maximum number of tokens for the LLM
# Default values for the output RSS feed
DEFAULT_OUTPUT_FEED_FILE = "summary_feed.xml"
DEFAULT_OUTPUT_FEED_TITLE = "Daily RSS Summary"
DEFAULT_OUTPUT_FEED_LINK = "http://localhost/summary_feed.xml" # Placeholder link
DEFAULT_OUTPUT_FEED_DESC = "A daily summary of articles from monitored RSS feeds."
# Default port for the HTTP server
DEFAULT_SERVER_PORT = 8000


# --- Global State ---
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
        print(f"Warning: Could not load processed IDs from {filename}: {e}. Starting fresh.")
        return set()

def save_processed_ids(ids_set, filename=PROCESSED_IDS_FILE):
    """Saves processed article IDs to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(list(ids_set), f) # Convert set to list for JSON serialization
    except IOError as e:
        print(f"Error saving processed IDs to {filename}: {e}")

def fetch_rss_feed(feed_url):
    """Fetches and parses the RSS feed."""
    print(f"Fetching RSS feed: {feed_url}")
    try:
        feed = feedparser.parse(feed_url)
        if feed.bozo:
            print(f"Warning: Feed may be malformed. Bozo reason: {feed.bozo_exception}")
        if not feed.entries:
            print("Warning: No entries found in the feed.")
        return feed.entries
    except Exception as e:
        print(f"Error fetching or parsing feed {feed_url}: {e}")
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
    # Ensure reasonable paragraph breaks
    cleaned = cleaned.replace(" . ", ". ")
    return cleaned

def fetch_article_content(url):
    """Fetches and extracts plain text content from an article URL."""
    print(f"  Fetching article content: {url}")
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
            print("  Warning: Specific article container not found, falling back to all <p> tags.")
            text_elements = soup.find_all('p')

        if not text_elements:
            print(f"  Warning: No text paragraphs found for {url}")
            return None

        # Join text, preserving some structure by adding newlines between elements
        content = "\n".join(p.get_text().strip() for p in text_elements if p.get_text().strip())
        return clean_text(content) # Clean the final joined text

    except requests.exceptions.RequestException as e:
        print(f"  Error fetching article {url}: {e}")
        return None
    except Exception as e:
        print(f"  Error parsing article {url}: {e}")
        return None

def summarize_text_with_langchain(text_to_summarize, llm):
    """Summarizes the given text using LangChain."""
    if not text_to_summarize:
         print("  Skipping summary for empty text.")
         return "Content unavailable to summarize."

    print("  Summarizing text...")

    # Check token count BEFORE creating prompt/chain
    # Estimate max tokens for the prompt itself
    prompt_overhead_estimate = 50
    available_tokens_for_text = MAX_TOKENS - prompt_overhead_estimate
    text_tokens = count_tokens(text_to_summarize)

    if text_tokens > available_tokens_for_text:
        print(f"  Warning: Text ({text_tokens} tokens) too long for model's {MAX_TOKENS} limit (estimated). Truncating...")
        # Simple truncation based on estimated character ratio
        # A more robust method might involve token-aware truncation
        estimated_chars_per_token = len(text_to_summarize) / text_tokens
        max_chars = int(available_tokens_for_text * estimated_chars_per_token * 0.95) # 5% buffer
        text_to_summarize = text_to_summarize[:max_chars]
        print(f"  Truncated text to ~{count_tokens(text_to_summarize)} tokens.")


    # Use a slightly more robust prompt
    prompt_template = """Please provide a concise summary of the following text. Focus on the key information and main points.

    TEXT:
    "{text}"

    CONCISE SUMMARY:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | llm | StrOutputParser()

    try:
        summary = chain.invoke({"text": text_to_summarize})
        print("  Summary generated.")
        # Basic post-processing: remove potential leading/trailing whitespace
        return summary.strip()
    except Exception as e:
        print(f"  Error during summarization: {e}")
        # Consider more specific error handling (e.g., context length exceeded)
        # You might want to check for specific error types from OpenAI/Langchain
        if "context_length_exceeded" in str(e).lower():
             return "Error: The combined text was too long to summarize, even after truncation."
        return "Error generating summary."

# --- Core Logic ---

def check_feed(feed_url, use_feed_summary):
    """Checks the feed for new articles and adds them to the buffer."""
    global processed_ids, new_articles_buffer
    print(f"\n[{datetime.datetime.now()}] Checking feed: {feed_url}")
    entries = fetch_rss_feed(feed_url)
    new_count = 0

    for entry in entries:
        article_id = entry.get('guid') or entry.get('id') or entry.get('link')
        if not article_id:
            print(f"  Warning: Skipping entry with no guid, id, or link: {entry.get('title', 'No Title')}")
            continue

        if article_id not in processed_ids:
            title = entry.get('title', 'No Title')
            link = entry.get('link')
            print(f"  Found new article: {title}")
            processed_ids.add(article_id)
            new_count += 1

            content_to_summarize = None
            if use_feed_summary:
                print("    Using summary/description from feed entry.")
                content_to_summarize = entry.get('summary') or entry.get('description')
                if not content_to_summarize:
                     print("    Warning: No summary/description found in feed entry.")
            elif link:
                 # Fetch content immediately if not using feed summary
                 content_to_summarize = fetch_article_content(link)
                 time.sleep(REQUEST_DELAY_SECONDS) # Delay even if fetch fails
                 if not content_to_summarize:
                      print(f"    Warning: Failed to fetch content for '{title}'.")
            else:
                 print(f"    Warning: No link found for '{title}' and not using feed summary.")


            # Buffer the entry and the content (even if None)
            new_articles_buffer.append((entry, content_to_summarize))
        # else: # No need for this else block now
            # print(f"  Skipping already processed article: {entry.get('title', 'No Title')}")
            # pass

    print(f"Finished checking feed. Found {new_count} new articles.")
    if new_count > 0:
         save_processed_ids(processed_ids)


def summarize_new_articles(llm, output_feed_file, output_feed_title, output_feed_link, output_feed_description):
    """Combines and summarizes articles, then adds the summary as a new entry to an RSS feed."""
    global processed_ids, new_articles_buffer
    print(f"\n[{datetime.datetime.now()}] Starting scheduled summarization and feed generation...")

    if not new_articles_buffer:
        print("No new articles in the buffer to summarize.")
        return

    print(f"Preparing summary for {len(new_articles_buffer)} new articles/entries...")
    articles_processed_this_run = set()
    combined_content_parts = []
    articles_included_count = 0
    first_article_link = None # To use as a fallback link for the summary entry

    buffer_copy = list(new_articles_buffer)
    new_articles_buffer.clear() # Clear global buffer now that we have a copy

    print("-" * 60)
    print("Articles included in this summary batch:")
    for entry, content_to_summarize in buffer_copy:
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
            print(f"  Warning: Could not determine a unique ID for article '{title}'. It may be reprocessed if it appears again.")

        if content_to_summarize:
            print(f"  - {title} ({link})")
            # Format for combining. Add more context if needed.
            formatted_article = f"Title: {title}\nLink: {link}\n\n{content_to_summarize}\n\n---\n\n"
            combined_content_parts.append(formatted_article)
            articles_included_count += 1
        else:
            # Still log the article title even if skipped for content
            print(f"  - Skipping content for '{title}' ({link}) - No content was fetched or found in feed.")

    combined_summary = "No summary generated (no content available)." # Default
    if combined_content_parts:
        print(f"\nGenerating combined summary for {articles_included_count} articles...")
        combined_content = "".join(combined_content_parts)
        # Simplified prompt prefix
        summary_prompt_prefix = f"Provide a concise combined summary of the following {articles_included_count} articles:\n\n"
        full_text_to_summarize = summary_prompt_prefix + combined_content

        # Generate the summary
        combined_summary = summarize_text_with_langchain(full_text_to_summarize, llm)

        # Print summary to console as well for logging/debugging
        print("\n--- Combined Daily Summary (for log) ---")
        print(combined_summary)
        print("--- End of Summary ---")
    else:
        print("\nNo content available from buffered articles to generate a combined summary.")
        # Decide if we should still create a feed entry (e.g., "No new content summarized today")
        # For now, we'll skip creating an entry if there was absolutely no content.
        print("Skipping RSS feed update as there was no content to summarize.")
        # Still need to mark articles as processed
        processed_ids.update(articles_processed_this_run)
        save_processed_ids(processed_ids)
        print(f"Finished scheduled run. Attempted to process {len(articles_processed_this_run)} unique article IDs. No feed generated.")
        return # Exit the function early

    # --- RSS Feed Generation ---
    print(f"\nGenerating RSS feed entry and updating {output_feed_file}...")
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
            # feedgen load_feed replaces existing metadata, so we load into a temp object
            temp_fg = FeedGenerator()
            temp_fg.load_feed(output_feed_file)
            # Transfer old entries to the new fg object
            for entry in temp_fg.entry():
                fg.add_entry(entry)
            print(f"  Loaded {len(fg.entry())} existing entries from feed file.")
        except Exception as e:
            print(f"  Warning: Could not load or parse existing feed file {output_feed_file}: {e}. Starting with a new feed.")
            # Proceed with the initialized fg object (new feed)
    else:
         print("  No existing feed file found. Creating a new feed.")


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
        print(f"  Successfully updated RSS feed: {output_feed_file}")
    except Exception as e:
        print(f"  Error saving RSS feed to {output_feed_file}: {e}")

    # Update the main processed set and save
    processed_ids.update(articles_processed_this_run)
    save_processed_ids(processed_ids)

    print("-" * 60)
    print(f"Finished scheduled summarization. Attempted to process {len(articles_processed_this_run)} unique article IDs.")
    if combined_content_parts:
         print(f"Successfully included content from {articles_included_count} articles in the combined summary generation.")

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
    print(f"[{datetime.datetime.now()}] Starting HTTP server on port {port} to serve files from the current directory.")
    print(f"[{datetime.datetime.now()}] Access the generated feed typically at http://localhost:{port}/{DEFAULT_OUTPUT_FEED_FILE}")

    with socketserver.TCPServer(("", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            # This might not be caught here if the main thread catches it first
            print("\nHTTP server received KeyboardInterrupt (likely from main thread). Shutting down...")
        finally:
            # Ensure cleanup happens
            httpd.server_close() # Close the server socket
            print(f"[{datetime.datetime.now()}] HTTP server on port {port} stopped.")

# --- Main Execution ---

def main():
    global processed_ids

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
        print(f"Error: Invalid summary_time format '{args.summary_time}'. Please use HH:MM (24-hour clock).")
        exit(1)


    processed_ids = load_processed_ids()
    print(f"Loaded {len(processed_ids)} processed article IDs from {PROCESSED_IDS_FILE}.")

    llm = ChatOpenAI(
        model_name=args.model,
        temperature=args.temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    print("\n--- RSS Summarizer Bot Started ---")
    print(f"Monitoring Feed: {args.feed_url}")
    print(f"Checking every: {args.check_interval} minutes")
    print(f"Daily Summary Time: {args.summary_time}")
    print(f"Using feed summary directly: {args.use_feed_summary}")
    print(f"LLM Model: {args.model}")
    print(f"Output Feed File: {args.output_feed_file}")
    print(f"Output Feed Title: {args.output_feed_title}")
    print(f"Output Feed Link: {args.output_feed_link}")
    print(f"Output Feed Description: {args.output_feed_description}")
    print(f"Serving feed on port: {args.port}") # Added log for port
    print("------------------------------------\n")

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
    print(f"[{datetime.datetime.now()}] Performing initial feed check...")
    check_feed(args.feed_url, args.use_feed_summary)

    # Define summary function with arguments bound using a dictionary
    # This makes it clearer which arguments are being passed
    summary_kwargs = {
        'llm': llm,
        'output_feed_file': args.output_feed_file,
        'output_feed_title': args.output_feed_title,
        'output_feed_link': args.output_feed_link,
        'output_feed_description': args.output_feed_description
    }
    bound_summarize_func = lambda: summarize_new_articles(**summary_kwargs)

    if args.run_once:
        print(f"\n[{datetime.datetime.now()}] Running summary immediately due to --run_once flag...")
        bound_summarize_func() # Call the bound function
        print(f"\n[{datetime.datetime.now()}] Run once complete. Feed generated and served at http://localhost:{args.port}/{args.output_feed_file}")
        # Keep running to serve the file until Ctrl+C if run_once was used
        print("Press Ctrl+C to stop serving.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
             print("\nExiting after run_once.")
        return

    # Schedule the feed check
    schedule.every(args.check_interval).minutes.do(check_feed, feed_url=args.feed_url, use_feed_summary=args.use_feed_summary)

    # Schedule the daily summary using the bound function
    schedule.every().day.at(args.summary_time).do(bound_summarize_func)

    print(f"\n[{datetime.datetime.now()}] Scheduler started. Waiting for the next check (every {args.check_interval} mins) or summary time ({args.summary_time})...")
    # Log the exact feed URL based on arguments
    print(f"[{datetime.datetime.now()}] Feed will be available at http://localhost:{args.port}/{args.output_feed_file}")

    # Main loop - now also implicitly keeps the server thread alive
    try:
        while True:
            schedule.run_pending()
            # Check if the server thread is still alive (optional)
            if not server_thread.is_alive():
                 print(f"[{datetime.datetime.now()}] Error: HTTP server thread has stopped unexpectedly. Attempting restart...")
                 # Attempt to restart the server thread
                 try:
                      server_thread = threading.Thread(target=run_http_server, args=(args.port,), daemon=True)
                      server_thread.start()
                      time.sleep(0.5) # Give it a moment
                      if not server_thread.is_alive(): # Check again
                           raise RuntimeError("Failed to restart server thread.")
                      print(f"[{datetime.datetime.now()}] HTTP server thread restarted successfully.")
                 except Exception as e:
                      print(f"[{datetime.datetime.now()}] CRITICAL ERROR: Failed to restart HTTP server thread: {e}. Exiting.")
                      # Depending on requirements, might want to exit or just log heavily
                      break # Exit the main loop

            time.sleep(60) # Check schedule every minute
    except KeyboardInterrupt:
         print(f"\n[{datetime.datetime.now()}] Ctrl+C received. Shutting down scheduler...")
         # Server thread is daemon, should exit. Explicit shutdown is handled in run_http_server.
    finally:
         # Any other cleanup if needed
         print(f"[{datetime.datetime.now()}] Main process exiting.")

if __name__ == "__main__":
    main()