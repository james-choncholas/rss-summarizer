import argparse
import datetime
import threading
import time
import os

# --- Configuration (Loads env vars, sets up logging, defines constants) ---
# Ensure config is imported first to set up logging and load environment variables
try:
    from config import (
        logger, OPENAI_API_KEY, CHECK_INTERVAL_MINUTES, SUMMARY_TIME,
        DEFAULT_OUTPUT_FEED_FILE, DEFAULT_OUTPUT_FEED_TITLE,
        DEFAULT_OUTPUT_FEED_DESC, DEFAULT_SERVER_PORT, PROCESSED_IDS_FILE,
        FEED_URLS, USE_FEED_SUMMARY, MODEL, TEMPERATURE, # Use FEED_URLS
        SYSTEM_PROMPT
    )
except ImportError as e:
    print(f"Error importing configuration: {e}")
    print("Ensure config.py exists and necessary environment variables (like OPENAI_API_KEY) are set.")
    exit(1)
except ValueError as e: # Catch specific error from config if API key is missing
    print(f"Configuration error: {e}")
    exit(1)


# --- Core Components ---
from utils import load_processed_ids
from langchain_openai import ChatOpenAI
from server import run_http_server
from scheduler import AppState, scheduled_check_feed_job, scheduled_summarize_job, run_scheduler, process_and_summarize

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Monitor RSS feeds, summarize new articles daily, publish to a new RSS feed, and serve it.")
    # Changed feed_url to be optional, defaulting to environment variable
    parser.add_argument("--feed_urls", type=lambda s: [url.strip() for url in s.split(',') if url.strip()], default=FEED_URLS, help="Comma-separated list of RSS feed URLs to monitor (can also be set via FEED_URLS env var).")
    # Use BooleanOptionalAction, default from config (env var)
    parser.add_argument("--use_feed_summary", default=USE_FEED_SUMMARY, action=argparse.BooleanOptionalAction, help="Use summary/description from feed entry directly instead of fetching full article content (can also be set via USE_FEED_SUMMARY env var: true/false).")
    # Default model from config (env var)
    parser.add_argument("--model", type=str, default=MODEL, help=f"OpenAI model name to use (default from MODEL env var or '{MODEL}').")
    # Default temperature from config (env var)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help=f"LLM temperature (creativity). Lower is more deterministic (default from TEMPERATURE env var or {TEMPERATURE}).")
    parser.add_argument("--check_interval", type=int, default=CHECK_INTERVAL_MINUTES, help=f"How often to check the feed in minutes (default: {CHECK_INTERVAL_MINUTES}).")
    parser.add_argument("--summary_time", type=str, default=SUMMARY_TIME, help=f"Time to run the daily summary in HH:MM format (24-hour clock) (default: {SUMMARY_TIME}).")
    parser.add_argument("--run_once", action="store_true", help="Run the check and summary once immediately, then exit (for testing).")
    # Added system prompt argument
    parser.add_argument("--system-prompt", type=str, default=SYSTEM_PROMPT, help="Override the system prompt for the summary.")
    # Output feed arguments
    parser.add_argument("--output_feed_file", type=str, default=DEFAULT_OUTPUT_FEED_FILE, help=f"Filename for the generated summary RSS feed (default: {DEFAULT_OUTPUT_FEED_FILE}).")
    parser.add_argument("--output_feed_title", type=str, default=DEFAULT_OUTPUT_FEED_TITLE, help=f"Title for the generated RSS feed (default: {DEFAULT_OUTPUT_FEED_TITLE}).")
    parser.add_argument("--output_feed_link", type=str, default=None, help="Base Link for the generated RSS feed (default: http://localhost:[port]/[output_file]).")
    parser.add_argument("--output_feed_description", type=str, default=DEFAULT_OUTPUT_FEED_DESC, help=f"Description for the generated RSS feed (default: {DEFAULT_OUTPUT_FEED_DESC}).")
    # Server argument
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT, help=f"Port to serve the RSS feed on (default: {DEFAULT_SERVER_PORT}).")
    # Added argument to specify serving directory
    parser.add_argument("--serve_dir", type=str, default=".", help="Directory to serve files from (default: current directory).")


    args = parser.parse_args()

    # --- Validate Arguments ---
    # Validate Feed URLs
    if not args.feed_urls:
        logger.error("Feed URLs must be provided either via the FEED_URLS environment variable (comma-separated) or the --feed_urls command-line argument.")
        exit(1)

    try:
        datetime.datetime.strptime(args.summary_time, '%H:%M')
    except ValueError:
        logger.error(f"Invalid summary_time format '{args.summary_time}'. Please use HH:MM (24-hour clock).")
        exit(1)

    if args.check_interval <= 0:
        logger.error(f"check_interval must be a positive number of minutes.")
        exit(1)

    # Construct the output feed link if not provided
    output_link = args.output_feed_link
    if output_link is None:
        # Ensure the filename doesn't start with / if we prepend http info
        feed_file_part = args.output_feed_file.lstrip('/')
        # Basic check if running locally or could be improved with hostname detection
        hostname = "localhost"
        output_link = f"http://{hostname}:{args.port}/{feed_file_part}"
        logger.info(f"Output feed link not specified, defaulting to: {output_link}")


    # --- Determine System Prompt ---
    system_prompt = args.system_prompt
    log_prompt_display = f"{system_prompt[:100]}..." if len(system_prompt) > 100 else system_prompt

    # --- Initialize Components ---
    logger.info("--- RSS Summarizer Bot Initializing ---")

    # Load initial processed IDs
    initial_processed_ids = load_processed_ids(PROCESSED_IDS_FILE)
    logger.info(f"Loaded {len(initial_processed_ids)} processed article IDs from {PROCESSED_IDS_FILE}.")

    # Initialize shared state
    app_state = AppState(
        processed_ids=initial_processed_ids,
        feed_urls=args.feed_urls, # Pass the list of URLs
        use_feed_summary=args.use_feed_summary, # Pass the global setting
        initial_check_interval=args.check_interval # Pass the base interval
    )

    # Initialize LLM
    try:
        llm = ChatOpenAI(
            model_name=args.model,
            temperature=args.temperature,
            openai_api_key=OPENAI_API_KEY
        )
        # Optional: Test LLM connection? Could add a simple invoke test here.
        logger.info(f"Initialized LLM: {args.model} (Temperature: {args.temperature})")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI LLM: {e}", exc_info=True)
        exit(1)

    # --- Log Configuration ---
    logger.info(f"Monitoring {len(args.feed_urls)} Feed(s):")
    for url in args.feed_urls:
        logger.info(f"  - {url}")
    logger.info(f"Checking every: {args.check_interval} minutes (initial interval)")
    logger.info(f"Daily Summary Time: {args.summary_time}")
    # Log the final effective value after parsing args
    logger.info(f"Using feed summary directly: {args.use_feed_summary}")
    logger.info(f"Using Model: {args.model}") # Log the model being used
    logger.info(f"Using Temperature: {args.temperature}") # Log the temperature
    logger.info(f"Output Feed File: {args.output_feed_file}")
    logger.info(f"Output Feed Title: {args.output_feed_title}")
    logger.info(f"Output Feed Link: {output_link}") # Use the final calculated link
    logger.info(f"Output Feed Description: {args.output_feed_description}")
    logger.info(f"Serving feed on port: {args.port}")
    logger.info(f"Serving files from directory: {os.path.abspath(args.serve_dir)}")
    logger.info(f"System prompt: {system_prompt}")
    logger.info("------------------------------------")


    # Prepare arguments for the summarization/feed generation part
    feed_generation_args = {
        'output_feed_file': args.output_feed_file,
        'output_feed_title': args.output_feed_title,
        'output_feed_link': output_link,
        'output_feed_description': args.output_feed_description,
    }

    # --- Start HTTP Server ---
    # Serve from the specified directory
    server_thread = threading.Thread(
        target=run_http_server,
        args=(args.port, args.serve_dir),
        daemon=True # Exit with main thread
    )
    server_thread.start()
    time.sleep(0.5) # Give server thread a moment to start/log


    # --- Handle --run_once ---
    if args.run_once:
        logger.info("Performing initial check (--run_once)...")
        for feed_url in args.feed_urls:
            logger.info(f"Checking feed (once): {feed_url}")
            # Call the check job for each feed (assuming updated signature)
            try:
                scheduled_check_feed_job(app_state, feed_url, args.use_feed_summary)
            except Exception as e:
                logger.error(f"Error checking feed {feed_url} during --run_once: {e}", exc_info=True)

        logger.info("Running summary immediately (--run_once)...")
        # We need to get the buffer contents after the check
        with app_state.lock:
            buffer_copy = list(app_state.articles_buffer)
            current_ids = app_state.processed_ids.copy()

        if buffer_copy:
             # Use the core processing logic directly
            updated_ids = process_and_summarize(
                articles_to_process=buffer_copy,
                processed_ids=current_ids,
                llm=llm,
                model_name=args.model,
                system_prompt=system_prompt,
                **feed_generation_args
            )
            # No need to update app_state here as we are exiting
            logger.info(f"Run once complete. Final processed IDs count: {len(updated_ids)}")
        else:
             logger.info("Run once complete. No new articles found to summarize.")


        logger.info(f"Feed generated (if new articles found) and served at {output_link}")
        logger.info("Press Ctrl+C to stop serving.")
        try:
            # Keep main thread alive only to keep server thread running
            while server_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
             logger.info("Exiting after --run_once.")
        return # Exit after run_once completes


    # --- Start Scheduler in a Background Thread ---
    # No explicit initial check here; the scheduler's first run will handle it.

    # Start the main scheduling loop in a separate thread
    scheduler_thread = threading.Thread(
        target=run_scheduler,
        args=(
            app_state, # Contains feed states (URLs, use_feed_summary, backoff info)
            args.check_interval, # Used as initial interval and maybe for logging
            args.summary_time,
            # Removed args.feed_urls
            # Removed args.use_feed_summary
            llm,
            args.model,
            feed_generation_args,
            system_prompt
        ),
        daemon=True # Allow main thread to exit even if scheduler has issues stopping
    )
    scheduler_thread.start()


    # --- Keep Main Thread Alive (and Monitor Server) ---
    logger.info(f"Scheduler thread started. Feed available at {output_link}")
    try:
        while True:
            if not server_thread.is_alive():
                 logger.error("HTTP server thread has stopped unexpectedly!")
                 # Optional: Attempt restart or exit cleanly
                 logger.info("Attempting to restart HTTP server...")
                 try:
                     server_thread = threading.Thread(target=run_http_server, args=(args.port, args.serve_dir), daemon=True)
                     server_thread.start()
                     time.sleep(1) # Give it time to start
                     if not server_thread.is_alive():
                         raise RuntimeError("Failed to restart server thread.")
                     logger.info("HTTP server thread restarted successfully.")
                 except Exception as e:
                     logger.critical(f"CRITICAL ERROR: Failed to restart HTTP server thread: {e}. Exiting.", exc_info=True)
                     break # Exit main loop

            if not scheduler_thread.is_alive():
                logger.error("Scheduler thread has stopped unexpectedly! Exiting.")
                break # Exit main loop

            time.sleep(5) # Check thread status periodically

    except KeyboardInterrupt:
         logger.info("Ctrl+C received in main thread. Shutting down...")
         # Scheduler loop handles its own KeyboardInterrupt.
         # Server thread is daemon and should exit.
    except Exception as e:
        logger.critical(f"Unhandled exception in main thread: {e}", exc_info=True)
    finally:
         # Any final cleanup if needed
         # Ensure server socket is closed (though run_http_server tries)
         logger.info("Main process exiting.")
         # We might want to explicitly signal threads to stop here if they weren't daemons
         # or if graceful shutdown is critical.


if __name__ == "__main__":
    main()