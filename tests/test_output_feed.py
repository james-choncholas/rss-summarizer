import os
import tempfile
import datetime
import pytest
from unittest.mock import patch, MagicMock, mock_open

import output_feed
from output_feed import MAX_FEED_ENTRIES

@pytest.fixture
def temp_feed_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_feed_file = os.path.join(tmpdir, "summary_feed.xml")
        extended_history_file = os.path.join(tmpdir, "extended_history_feed.xml")
        yield output_feed_file, extended_history_file

@patch("output_feed.logger")
@patch("output_feed.feedparser.parse")
@patch("output_feed.FeedGenerator")
def test_generate_new_feed_no_history(mock_fg, mock_parse, mock_logger, temp_feed_files):
    output_feed_file, extended_history_file = temp_feed_files
    mock_parse.return_value = MagicMock(entries=[], bozo=False)
    fg_instance = MagicMock()
    mock_fg.return_value = fg_instance

    output_feed.generate_summary_feed(
        output_feed_file=output_feed_file,
        output_feed_title="Test Feed",
        output_feed_link="http://example.com",
        output_feed_description="A test feed",
        combined_summary="Test summary content.",
        extended_history_file=extended_history_file
    )

    # Should create a new entry and call rss_file for both files
    assert fg_instance.rss_file.call_count == 2
    fg_instance.add_entry.assert_called()
    fg_instance.title.assert_any_call("Test Feed")
    fg_instance.link.assert_any_call(href="http://example.com", rel='alternate')
    fg_instance.description.assert_any_call("A test feed")
    fg_instance.language.assert_any_call('en')

@patch("output_feed.logger")
@patch("output_feed.feedparser.parse")
@patch("output_feed.FeedGenerator")
@patch("output_feed.os.path.exists", return_value=True)
def test_generate_feed_with_existing_history(mock_os_exists, mock_fg, mock_parse, mock_logger, temp_feed_files):
    output_feed_file, extended_history_file = temp_feed_files
    # Simulate existing entries
    old_entry = {
        'title': 'Old Summary',
        'id': 'urn:summary:20240510:Test_Feed',
        'published': datetime.datetime(2024, 5, 10, tzinfo=datetime.timezone.utc),
        'updated': datetime.datetime(2024, 5, 10, tzinfo=datetime.timezone.utc),
        'link': 'http://example.com',
        'content': 'Old summary content.'
    }
    mock_parse.return_value = MagicMock(entries=[old_entry], bozo=False)
    fg_instance = MagicMock()
    mock_fg.return_value = fg_instance

    output_feed.generate_summary_feed(
        output_feed_file=output_feed_file,
        output_feed_title="Test Feed",
        output_feed_link="http://example.com",
        output_feed_description="A test feed",
        combined_summary="New summary content.",
        extended_history_file=extended_history_file
    )

    # Should prepend new entry and keep old one
    assert fg_instance.add_entry.call_count >= 2
    fg_instance.rss_file.assert_called()

@patch("output_feed.logger")
@patch("output_feed.feedparser.parse")
@patch("output_feed.FeedGenerator")
@patch("output_feed.os.path.exists", return_value=True)
def test_generate_feed_truncates_to_max(mock_os_exists, mock_fg, mock_parse, mock_logger, temp_feed_files):
    output_feed_file, extended_history_file = temp_feed_files
    # Simulate more than MAX_FEED_ENTRIES old entries
    old_entries = [
        {
            'title': f'Old {i}',
            'id': f'id-{i}',
            'published': datetime.datetime(2024, 5, i+1, tzinfo=datetime.timezone.utc),
            'updated': datetime.datetime(2024, 5, i+1, tzinfo=datetime.timezone.utc),
            'link': 'http://example.com',
            'content': f'Content {i}'
        } for i in range(MAX_FEED_ENTRIES + 5)
    ]
    mock_parse.return_value = MagicMock(entries=old_entries, bozo=False)
    fg_output = MagicMock()
    fg_history = MagicMock()
    mock_fg.side_effect = [fg_output, fg_history]  # First for output, second for history

    output_feed.generate_summary_feed(
        output_feed_file=output_feed_file,
        output_feed_title="Test Feed",
        output_feed_link="http://example.com",
        output_feed_description="A test feed",
        combined_summary="Newest summary.",
        extended_history_file=extended_history_file
    )

    # Only MAX_FEED_ENTRIES entries should be added to the output feed
    assert fg_output.add_entry.call_count == MAX_FEED_ENTRIES
    # All entries (old ones + 1 new one) should be added to the history feed
    assert fg_history.add_entry.call_count == len(old_entries) + 1
    fg_output.rss_file.assert_called()
    fg_history.rss_file.assert_called()

@patch("output_feed.logger")
@patch("output_feed.feedparser.parse", side_effect=Exception("Parse error"))
@patch("output_feed.FeedGenerator")
@patch("output_feed.os.path.exists", return_value=True)
def test_generate_feed_handles_parse_error(mock_os_exists, mock_fg, mock_parse, mock_logger, temp_feed_files):
    output_feed_file, extended_history_file = temp_feed_files
    fg_instance = MagicMock()
    mock_fg.return_value = fg_instance

    output_feed.generate_summary_feed(
        output_feed_file=output_feed_file,
        output_feed_title="Test Feed",
        output_feed_link="http://example.com",
        output_feed_description="A test feed",
        combined_summary="Summary after error.",
        extended_history_file=extended_history_file
    )

    # Should still create a feed even if parse fails
    fg_instance.rss_file.assert_called()
    # Check that the logger was warned about the parse error specifically.
    mock_logger.warning.assert_any_call(
        f"  Warning: Could not load or parse extended history file {extended_history_file}: Parse error. Starting with a new history.",
        exc_info=True
    )

@patch("output_feed.logger")
@patch("output_feed.feedparser.parse")
@patch("output_feed.FeedGenerator")
def test_generate_feed_write_error(mock_fg, mock_parse, mock_logger, temp_feed_files):
    output_feed_file, extended_history_file = temp_feed_files
    mock_parse.return_value = MagicMock(entries=[], bozo=False)
    fg_instance = MagicMock()
    fg_instance.rss_file.side_effect = Exception("Write failed!")
    mock_fg.return_value = fg_instance

    output_feed.generate_summary_feed(
        output_feed_file=output_feed_file,
        output_feed_title="Test Feed",
        output_feed_link="http://example.com",
        output_feed_description="A test feed",
        combined_summary="Summary for write error.",
        extended_history_file=extended_history_file
    )

    # Should log error for write failure
    mock_logger.error.assert_any_call(
        f"  Error saving extended history file to {extended_history_file}: Write failed!"
    )
    # Should also try to write the output feed and log error
    assert mock_logger.error.call_count >= 1

@patch("output_feed.logger")
@patch("output_feed.feedparser.parse")
@patch("output_feed.FeedGenerator")
def test_generate_feed_no_summary(mock_fg, mock_parse, mock_logger, temp_feed_files):
    """Test that a new entry is not created when the combined_summary is None."""
    output_feed_file, extended_history_file = temp_feed_files
    mock_parse.return_value = MagicMock(entries=[], bozo=False)
    fg_instance = MagicMock()
    mock_fg.return_value = fg_instance

    output_feed.generate_summary_feed(
        output_feed_file=output_feed_file,
        output_feed_title="Test Feed",
        output_feed_link="http://example.com",
        output_feed_description="A test feed",
        combined_summary=None,
        extended_history_file=extended_history_file
    )

    # If summary is None, no new entry should be added.
    fg_instance.add_entry.assert_not_called()