import logging
import os
from dotenv import load_dotenv

# --- Configuration & Constants ---
load_dotenv() # Load environment variables first

# --- Environment & API ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Use logger here once it's configured, or raise error immediately
    # For now, raise error as logger isn't ready yet.
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in the .env file.")

_feed_urls_str = os.getenv("FEED_URLS")
if _feed_urls_str:
    # Split by comma and strip whitespace from each URL
    FEED_URLS = [url.strip() for url in _feed_urls_str.split(',') if url.strip()]
else:
    FEED_URLS = [] # Default to an empty list if not set

# --- Behavior ---
REQUEST_DELAY_SECONDS = 1  # Delay between fetching full articles
REQUEST_TIMEOUT_SECONDS = 15 # Timeout for fetching article content
USER_AGENT = "RSSSummarizerBot/1.0 (+https://github.com/your-repo/rss-summarizer)" # User agent for requests
CHECK_INTERVAL_MINUTES = 30 # How often to check the feed (in minutes)
SUMMARY_TIME = os.getenv("SUMMARY_TIME", "8:00") # Time to run the daily summary (HH:MM)
MAX_TOKENS = 4096 # Max tokens for the LLM input (including prompt)
DEFAULT_SYSTEM_PROMPT_TEMPLATE = "Provide a concise combined summary of the following articles:\n\n"

# -- Load behavior settings from environment with defaults --
# USE_FEED_SUMMARY (Boolean: true/1/yes/y -> True, otherwise False)
_use_feed_summary_str = os.getenv("USE_FEED_SUMMARY", "false")
USE_FEED_SUMMARY = _use_feed_summary_str.lower() in ('true', '1', 't', 'yes', 'y')

# MODEL (String)
MODEL = os.getenv("MODEL", "gpt-3.5-turbo")

# TEMPERATURE (Float)
_default_temp = 0.3
try:
    TEMPERATURE = float(os.getenv("TEMPERATURE", str(_default_temp)))
except (ValueError, TypeError):
    # Use logger once available below
    print(f"Warning: Invalid TEMPERATURE environment variable. Using default: {_default_temp}") # Temporary print
    TEMPERATURE = _default_temp

# --- Files & Paths ---
PROCESSED_IDS_FILE = "processed_article_ids.json" # File to store processed article IDs

# --- Output Feed Defaults ---
DEFAULT_OUTPUT_FEED_FILE = "summary_feed.xml"
DEFAULT_OUTPUT_FEED_TITLE = "Daily RSS Summary"
# Default link will be determined later based on the actual port
# DEFAULT_OUTPUT_FEED_LINK = "http://localhost:8000/summary_feed.xml"
DEFAULT_OUTPUT_FEED_DESC = "A daily summary of articles from monitored RSS feeds."

# --- Server ---
DEFAULT_SERVER_PORT = 8000

# --- Logging Setup ---
# Configure logging globally here
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log temperature conversion warning properly now that logger is configured
if 'TEMPERATURE' in os.environ:
    try:
        float(os.environ['TEMPERATURE'])
    except (ValueError, TypeError):
         logger.warning(f"Invalid TEMPERATURE environment variable '{os.environ['TEMPERATURE']}'. Using default: {TEMPERATURE}")

# Optionally, you could validate configurations here
# Example:
# try:
#     CHECK_INTERVAL_MINUTES = int(CHECK_INTERVAL_MINUTES)
# except (TypeError, ValueError):
#     logger.warning(f"Invalid CHECK_INTERVAL_MINUTES value. Using default: {30}")
#     CHECK_INTERVAL_MINUTES = 30 