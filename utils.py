import json
import logging
import os

# Import logger from config (assuming config.py is in the same directory)
from config import logger, PROCESSED_IDS_FILE

def load_processed_ids(filename=PROCESSED_IDS_FILE):
    """Loads processed article IDs from a JSON file."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return set(json.load(f))
        else:
            logger.info(f"Processed IDs file '{filename}' not found. Starting with an empty set.")
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