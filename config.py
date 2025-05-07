import os
import logging
from dotenv import load_dotenv
import sys

# --- Load Environment Variables ---
load_dotenv() # Load variables from .env file

# --- Logging Setup ---
# Basic configuration - logs INFO and higher to stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    stream=sys.stdout)

# Get a logger instance for the application
logger = logging.getLogger("RSSSummarizer")

# --- Core Configuration ---

# API Configuration
# Renamed from OPENAI_API_KEY for generality
API_KEY = os.getenv("API_KEY")
# Added API_URL for compatibility with other OpenAI-like APIs
API_URL = os.getenv("API_URL", None) # Default to None, ChatOpenAI uses default OpenAI URL if None
API_MODEL = os.getenv("API_MODEL", "gpt-4o-mini") # Default model
temperature_str = os.getenv("TEMPERATURE", "0.3") # Default temperature
try:
    TEMPERATURE = float(temperature_str)
except ValueError:
    logger.warning(f"Invalid temperature value: {temperature_str}. Using default of 0.3.")
    TEMPERATURE = 0.3

# --- Feed Configuration ---
# Fetch URLs from environment variable, split by comma, strip whitespace
feed_urls_str = os.getenv("FEED_URLS", "") # Default to empty string if not set
FEED_URLS = [url.strip() for url in feed_urls_str.split(',') if url.strip()]
# Control whether to use feed's summary or fetch full content from link.
use_feed_summary_str = os.getenv("USE_FEED_SUMMARY", "false")
USE_FEED_SUMMARY = use_feed_summary_str.lower() in ('true', '1', 't', 'yes', 'y')

REQUEST_DELAY_SECONDS = 1  # Delay between fetching full articles
REQUEST_TIMEOUT_SECONDS = 15 # Timeout for fetching article content
USER_AGENT = "RSSSummarizerBot/1.0 (+https://github.com/your-repo/rss-summarizer)" # User agent for requests

# --- Scheduler Configuration ---
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", 30)) # How often to check feeds
SUMMARY_TIME = os.getenv("SUMMARY_TIME", "08:00") # Time for daily summary (HH:MM)


# --- Output Feed Configuration ---
DEFAULT_OUTPUT_FEED_FILE = os.getenv("OUTPUT_FEED_FILE", "output/summary_feed.xml")
DEFAULT_OUTPUT_FEED_TITLE = os.getenv("OUTPUT_FEED_TITLE", "Daily Summarized Feed")
DEFAULT_OUTPUT_FEED_DESC = os.getenv("OUTPUT_FEED_DESC", "Summaries of articles from monitored feeds.")
DEFAULT_SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))

# --- Persistence ---
PROCESSED_IDS_FILE = os.getenv("PROCESSED_IDS_FILE", "data/processed_ids.json")

# --- Prompting ---
DEFAULT_MAX_TOKENS = 4096 # Max tokens for the LLM input (including prompt)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", DEFAULT_MAX_TOKENS)) # Max tokens for the LLM input
# Default system prompt - loaded from env var or defaults to this string
DEFAULT_SYSTEM_PROMPT = """You are an expert summarizer. Given the following article text, provide a concise summary (around 2-4 sentences) focusing on the key information. Respond ONLY with the summary text, no preamble or extra phrases like "Here is the summary:". Be objective and accurate."""
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)


# --- Validation (Optional but recommended) ---
# Example validation: Check if API key is present
if not API_KEY:
    # Use logger instead of print for consistency
    logger.error("API_KEY environment variable not set.")
    # Raise ValueError to be caught in main.py for graceful exit
    raise ValueError("API_KEY environment variable not set.")

# Ensure data and output directories exist
os.makedirs(os.path.dirname(PROCESSED_IDS_FILE), exist_ok=True)
os.makedirs(os.path.dirname(DEFAULT_OUTPUT_FEED_FILE), exist_ok=True)

logger.info("Configuration loaded successfully.")