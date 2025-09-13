import pytest
import os
from unittest import mock
import importlib

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# We need to reload the config module for each test to isolate environment changes
import config # Initial import to locate the module

@pytest.fixture(autouse=True)
def reload_config():
    """Fixture to reload the config module before each test."""
    # Revert fixture changes - we'll use setattr/delattr or monkeypatch
    importlib.reload(config)

def test_api_key_present(monkeypatch):
    """Test that API_KEY is loaded correctly."""
    monkeypatch.setenv("API_KEY", "test_key_123")
    importlib.reload(config)
    assert config.API_KEY == "test_key_123"

def test_api_key_missing(monkeypatch):
    """Test that ValueError is raised if API_KEY is not set."""
    # Temporarily remove the API_KEY to test the validation
    if "API_KEY" in os.environ:
        monkeypatch.delenv("API_KEY", raising=False)

    # Mock load_dotenv to prevent it from loading a .env file
    with mock.patch('dotenv.load_dotenv'):
        # The ValueError is raised when the module is loaded, so we need to reload it
        with pytest.raises(ValueError, match="API_KEY environment variable not set."):
            importlib.reload(config)

    # Restore the API_KEY for other tests
    monkeypatch.setenv("API_KEY", "dummy_key_for_testing")


def test_feed_urls_parsing(monkeypatch):
    """Test parsing of FEED_URLS from environment variable."""
    monkeypatch.setenv("FEED_URLS", "http://url1.com , http://url2.com/feed,http://url3.com ")
    importlib.reload(config)
    assert config.FEED_URLS == ["http://url1.com", "http://url2.com/feed", "http://url3.com"]

def test_feed_urls_empty_string(monkeypatch):
    """Test FEED_URLS is empty list when env var is empty string."""
    monkeypatch.setenv("FEED_URLS", "")
    importlib.reload(config)
    assert config.FEED_URLS == []

def test_use_feed_summary_true_values(monkeypatch):
    """Test USE_FEED_SUMMARY evaluates to True for various 'true' strings."""
    for val in ['true', '1', 't', 'yes', 'y', 'TRUE', 'YES']:
        monkeypatch.setenv("USE_FEED_SUMMARY", val)
        importlib.reload(config)
        assert config.USE_FEED_SUMMARY is True

def test_use_feed_summary_false_values(monkeypatch):
    """Test USE_FEED_SUMMARY evaluates to False for various 'false' strings."""
    for val in ['false', '0', 'f', 'no', 'n', 'FALSE', 'NO', 'random_string', '']:
        monkeypatch.setenv("USE_FEED_SUMMARY", val)
        importlib.reload(config)
        assert config.USE_FEED_SUMMARY is False

def test_model_loading(monkeypatch):
    """Test loading API_MODEL from environment and its default."""
    # Test default
    if "API_MODEL" in os.environ:
        monkeypatch.delenv("API_MODEL")
    importlib.reload(config)
    assert config.API_MODEL == "gpt-4o-mini"

    # Test custom value
    monkeypatch.setenv("API_MODEL", "gpt-4")
    importlib.reload(config)
    assert config.API_MODEL == "gpt-4"

def test_temperature_loading(monkeypatch):
    """Test loading TEMPERATURE from environment, its default, and error handling."""
    default_temp = 0.3
    # Test default
    if "TEMPERATURE" in os.environ:
        monkeypatch.delenv("TEMPERATURE")
    importlib.reload(config)
    assert config.TEMPERATURE == default_temp

    # Test valid custom float
    monkeypatch.setenv("TEMPERATURE", "0.7")
    importlib.reload(config)
    assert config.TEMPERATURE == 0.7

    # Test invalid custom value
    monkeypatch.setenv("TEMPERATURE", "invalid-temp")
    importlib.reload(config)
    assert config.TEMPERATURE == default_temp

def test_request_delay_seconds(monkeypatch):
    monkeypatch.setenv("REQUEST_DELAY_SECONDS", "5")
    importlib.reload(config)
    assert config.REQUEST_DELAY_SECONDS == 5

def test_request_timeout_seconds(monkeypatch):
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "20")
    importlib.reload(config)
    assert config.REQUEST_TIMEOUT_SECONDS == 20

def test_check_interval_minutes(monkeypatch):
    monkeypatch.setenv("CHECK_INTERVAL_MINUTES", "60")
    importlib.reload(config)
    assert config.CHECK_INTERVAL_MINUTES == 60

def test_summary_time(monkeypatch):
    monkeypatch.setenv("SUMMARY_TIME", "10:00")
    importlib.reload(config)
    assert config.SUMMARY_TIME == "10:00"

def test_max_tokens(monkeypatch):
    monkeypatch.setenv("MAX_TOKENS", "8192")
    importlib.reload(config)
    assert config.MAX_TOKENS == 8192

def test_processed_ids_file(monkeypatch, tmp_path):
    # Create a temporary directory for the test
    d = tmp_path / "data"
    d.mkdir()
    test_file = d / "test_ids.json"
    
    monkeypatch.setenv("PROCESSED_IDS_FILE", str(test_file))
    importlib.reload(config)
    assert config.PROCESSED_IDS_FILE == str(test_file)
