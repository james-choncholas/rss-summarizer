import os
import tempfile
import datetime
import pytest
from unittest.mock import patch, MagicMock, mock_open

import output_feed

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
def test_generate_feed_truncates_to_10(mock_os_exists, mock_fg, mock_parse, mock_logger, temp_feed_files):
    output_feed_file, extended_history_file = temp_feed_files
    # Simulate 15 old entries
    old_entries = [
        {
            'title': f'Old {i}',
            'id': f'id-{i}',
            'published': datetime.datetime(2024, 5, i+1, tzinfo=datetime.timezone.utc),
            'updated': datetime.datetime(2024, 5, i+1, tzinfo=datetime.timezone.utc),
            'link': 'http://example.com',
            'content': f'Content {i}'
        } for i in range(15)
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

    # Only 10 entries should be added to the output feed (the first FeedGenerator instance)
    # A new entry plus the first 9 from the old_entries should make 10 total
    assert fg_output.add_entry.call_count == 10
    # All entries (15 old ones + 1 new one) should be added to the history feed
    assert fg_history.add_entry.call_count == 16
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
    # Check logger.warning was called with the expected message and exc_info=True (positional or keyword)
    found = False
    expected_msg = f"  Warning: Could not load or parse extended history file {extended_history_file}: Parse error. Starting with a new history."
    for call in mock_logger.warning.call_args_list:
        args, kwargs = call
        if args and expected_msg in args[0] and (len(args) > 1 and args[1] == True or kwargs.get('exc_info') is True):
            found = True
            break
    assert found, "Expected warning log for parse error not found."

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
