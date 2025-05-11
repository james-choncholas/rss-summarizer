import pytest
import datetime as dt
from unittest.mock import Mock, patch, call
from freezegun import freeze_time

# Assuming your scheduler code is in 'scheduler.py'
from scheduler import (
    FeedState,
    AppState,
    process_and_summarize,
    scheduled_check_feed_job,
    scheduled_summarize_job,
    INITIAL_BACKOFF_MINUTES,
    MAX_BACKOFF_MINUTES,
    BACKOFF_FACTOR,
    JITTER_FACTOR
)

# --- Fixtures ---

@pytest.fixture
def initial_timestamp():
    return dt.datetime(2023, 10, 27, 10, 0, 0)

@pytest.fixture
def feed_state(initial_timestamp):
    """Provides a default FeedState instance."""
    with freeze_time(initial_timestamp):
        return FeedState(url="http://example.com/rss", use_feed_summary=False)

@pytest.fixture
def app_state(initial_timestamp):
    """Provides a default AppState instance."""
    with freeze_time(initial_timestamp):
        return AppState(
            processed_ids={"id1", "id2"},
            feed_urls=["http://example.com/rss", "http://another.com/feed"],
            use_feed_summary=False,
            initial_check_interval=60
        )

# --- Tests for FeedState ---

def test_feed_state_initialization(initial_timestamp):
    """Test FeedState default values upon initialization."""
    url = "http://test.com"
    with freeze_time(initial_timestamp):
        state = FeedState(url=url, use_feed_summary=True)
        assert state.url == url
        assert state.use_feed_summary is True
        assert state.last_check_time == dt.datetime.min
        # Explicitly set next_check_time for direct FeedState creation in tests
        state.next_check_time = initial_timestamp
        assert state.next_check_time == initial_timestamp # Should be 'now' due to freeze_time
        assert state.failure_count == 0
        assert state.current_backoff_minutes == INITIAL_BACKOFF_MINUTES
        assert state.is_checking is False

# --- Tests for AppState ---

def test_app_state_initialization(initial_timestamp):
    """Test AppState initialization and feed_states creation."""
    processed = {"id1"}
    urls = ["http://feed1.com", "http://feed2.com"]
    initial_interval = 30
    with freeze_time(initial_timestamp):
        state = AppState(
            processed_ids=processed,
            feed_urls=urls,
            use_feed_summary=False,
            initial_check_interval=initial_interval
        )
        assert state.processed_ids == processed
        assert state.articles_buffer == []
        assert state.initial_check_interval == float(initial_interval)
        assert len(state.feed_states) == len(urls)
        assert "http://feed1.com" in state.feed_states
        assert "http://feed2.com" in state.feed_states

        # Check one of the FeedState objects within AppState
        feed1_state = state.feed_states["http://feed1.com"]
        assert isinstance(feed1_state, FeedState)
        assert feed1_state.url == "http://feed1.com"
        assert feed1_state.use_feed_summary is False
        assert feed1_state.next_check_time == initial_timestamp # Should be 'now' due to freeze_time
        assert feed1_state.current_backoff_minutes == float(initial_interval)

# --- Add more tests below ---

# --- Tests for process_and_summarize ---

@patch('scheduler.summarize_text_with_langchain')
@patch('scheduler.generate_summary_feed')
@patch('scheduler.save_processed_ids')
@patch('scheduler.logger') # Mock logger to prevent console output during tests
def test_process_and_summarize_no_articles(
    mock_logger, mock_save_ids, mock_generate_feed, mock_summarize, app_state):
    """Test processing when the input list is empty."""
    articles_to_process = []
    initial_ids = app_state.processed_ids.copy()
    llm_mock = Mock()
    feed_args = {
        "output_feed_file": "test_feed.xml",
        "output_feed_title": "Test Title",
        "output_feed_link": "http://example.com",
        "output_feed_description": "Test Desc",
        "extended_history_file": "test_history.xml",
    }

    updated_ids = process_and_summarize(
        articles_to_process=articles_to_process,
        processed_ids=app_state.processed_ids,
        llm=llm_mock,
        model_name="test-model",
        system_prompt="Test System Prompt",
        **feed_args
    )

    assert updated_ids == initial_ids # IDs should be unchanged
    mock_summarize.assert_not_called()
    # generate_summary_feed *is* called even with no articles, to create an empty feed or placeholder
    mock_generate_feed.assert_called_once()
    # save_processed_ids *is* called to ensure persistence even if no new articles
    mock_save_ids.assert_called_once_with(initial_ids)
    mock_logger.info.assert_any_call("No new articles in the buffer to summarize.")

@patch('scheduler.summarize_text_with_langchain')
@patch('scheduler.generate_summary_feed')
@patch('scheduler.save_processed_ids')
@patch('scheduler.logger')
def test_process_and_summarize_with_articles(
    mock_logger, mock_save_ids, mock_generate_feed, mock_summarize, app_state):
    """Test processing with a list of articles."""
    entry1 = {'title': 'Article 1', 'link': 'http://link1.com', 'guid': 'guid1'}
    entry2 = {'title': 'Article 2', 'link': 'http://link2.com', 'id': 'id2'} # Use 'id' as guid alternative
    entry3 = {'title': 'Article 3', 'link': 'http://link3.com'} # Use link as id
    entry4 = {'title': 'Article 4', 'link': 'http://link4.com', 'guid': 'guid4'} # No content
    articles_to_process = [
        (entry1, "Content 1", "http://feedA.com"),
        (entry2, "Content 2", "http://feedB.com"),
        (entry3, "Content 3", "http://feedA.com"),
        (entry4, None, "http://feedC.com"), # Article with no content
    ]
    initial_ids = app_state.processed_ids.copy() # Contains {"id1", "id2"}
    expected_ids_after_processing = initial_ids.union({"guid1", "http://link3.com", "guid4"}) # 'id2' was already present
    mock_summary = "This is the combined summary."
    mock_summarize.return_value = mock_summary
    llm_mock = Mock()
    feed_args = {
        "output_feed_file": "test_feed.xml",
        "output_feed_title": "Test Title",
        "output_feed_link": "http://example.com",
        "output_feed_description": "Test Desc",
        "extended_history_file": "test_history.xml",
    }

    updated_ids = process_and_summarize(
        articles_to_process=articles_to_process,
        processed_ids=initial_ids,
        llm=llm_mock,
        model_name="test-model",
        system_prompt="Test System Prompt",
        **feed_args
    )

    # --- Assertions ---
    assert updated_ids == expected_ids_after_processing

    # Check summarization call
    mock_summarize.assert_called_once()
    call_args, _ = mock_summarize.call_args
    text_to_summarize = call_args[0]
    assert text_to_summarize.startswith("Test System Prompt")
    assert "Title: Article 1" in text_to_summarize
    assert "Link: http://link1.com" in text_to_summarize
    assert "Content 1" in text_to_summarize
    assert "Title: Article 2" in text_to_summarize
    assert "Content 2" in text_to_summarize
    assert "Title: Article 3" in text_to_summarize
    assert "Content 3" in text_to_summarize
    assert "Title: Article 4" not in text_to_summarize # Skipped due to no content
    assert call_args[1] == llm_mock
    assert call_args[2] == "test-model"

    # Check feed generation call
    mock_generate_feed.assert_called_once_with(
        output_feed_file="test_feed.xml",
        output_feed_title="Test Title",
        output_feed_link="http://example.com",
        output_feed_description="Test Desc",
        combined_summary=mock_summary,
        extended_history_file="test_history.xml",
    )

    # Check processed IDs saving
    mock_save_ids.assert_called_once_with(expected_ids_after_processing)

    # Check logging (optional, but good for sanity)
    mock_logger.info.assert_any_call("Preparing summary for 4 new articles/entries...")
    mock_logger.info.assert_any_call("  - [http://feedA.com] Article 1 (http://link1.com)")
    mock_logger.info.assert_any_call("  - [http://feedB.com] Article 2 (http://link2.com)")
    mock_logger.info.assert_any_call("  - [http://feedA.com] Article 3 (http://link3.com)")
    mock_logger.info.assert_any_call("  - Skipping content for \'Article 4\' (http://link4.com) - No content available.")
    mock_logger.info.assert_any_call("Generating combined summary for 3 articles...")
    mock_logger.info.assert_any_call(f"Finished scheduled summarization. Processed IDs in this batch: {len({'guid1', 'id2', 'http://link3.com', 'guid4'})}.") # id2 was already known but processed now
    mock_logger.info.assert_any_call(f"Total unique processed IDs known: {len(expected_ids_after_processing)}.")
    mock_logger.info.assert_any_call("Included content from 3 articles from 3 feeds in the summary generation.")

# --- Tests for scheduled_check_feed_job ---

@patch('scheduler.check_feed')
@patch('scheduler.logger')
@patch('scheduler.random')
def test_scheduled_check_feed_job_success(
    mock_random, mock_logger, mock_check_feed, app_state, initial_timestamp):
    """Test scheduled_check_feed_job on successful feed fetch."""
    # --- Setup ---
    feed_url = "http://example.com/rss"
    feed_state = app_state.feed_states[feed_url]
    feed_state.current_backoff_minutes = 120
    feed_state.failure_count = 2
    initial_buffer_len = len(app_state.articles_buffer)
    initial_ids = app_state.processed_ids.copy()

    new_entry1 = {'title': 'New Article 1', 'link': 'newlink1', 'guid': 'newguid1'}
    new_entry2 = {'title': 'New Article 2', 'link': 'newlink2', 'id': 'newid2'}
    new_entries_tuples = [(new_entry1, "New Content 1"), (new_entry2, "New Content 2")]
    updated_ids_from_check = initial_ids.union({"newguid1", "newid2"})
    mock_check_feed.return_value = (new_entries_tuples, updated_ids_from_check)

    # Define the time *inside* the freeze_time context
    frozen_now = initial_timestamp + dt.timedelta(minutes=1)
    with freeze_time(frozen_now):
        mock_random.uniform.return_value = 0

        # --- Action ---
        scheduled_check_feed_job(app_state, feed_state)

        # --- Assertions ---
        mock_check_feed.assert_called_once_with(feed_url, feed_state.use_feed_summary, initial_ids)

        assert len(app_state.articles_buffer) == initial_buffer_len + len(new_entries_tuples)
        assert app_state.articles_buffer[-1] == (new_entry2, "New Content 2", feed_url)
        assert app_state.processed_ids == updated_ids_from_check

        assert feed_state.is_checking is False
        assert feed_state.failure_count == 0
        assert feed_state.last_check_time == frozen_now # Compare against the frozen time
        assert feed_state.current_backoff_minutes == app_state.initial_check_interval
        expected_next_check = frozen_now + dt.timedelta(minutes=app_state.initial_check_interval)
        assert feed_state.next_check_time == expected_next_check
        mock_logger.info.assert_any_call(f"Successfully checked {feed_url}. Added {len(new_entries_tuples)} new items. Next check in ~{app_state.initial_check_interval:.1f} min.")

@patch('scheduler.check_feed')
@patch('scheduler.logger')
@patch('scheduler.random')
def test_scheduled_check_feed_job_failure_and_backoff(
    mock_random, mock_logger, mock_check_feed, app_state, initial_timestamp):
    """Test scheduled_check_feed_job on failed feed fetch and backoff calculation."""
    # --- Setup ---
    feed_url = "http://another.com/feed"
    feed_state = app_state.feed_states[feed_url]
    initial_failure_count = feed_state.failure_count
    initial_backoff = feed_state.current_backoff_minutes
    initial_buffer_len = len(app_state.articles_buffer)
    initial_ids = app_state.processed_ids.copy()

    mock_check_feed.side_effect = Exception("Fetch failed!")

    # Define the time *inside* the freeze_time context
    frozen_now = initial_timestamp + dt.timedelta(minutes=1)
    with freeze_time(frozen_now):
        mock_random.uniform.return_value = 0.1 # Apply 10% positive jitter

        # --- Action ---
        scheduled_check_feed_job(app_state, feed_state)

        # --- Assertions ---
        mock_check_feed.assert_called_once_with(feed_url, feed_state.use_feed_summary, initial_ids)

        assert len(app_state.articles_buffer) == initial_buffer_len
        assert app_state.processed_ids == initial_ids

        assert feed_state.is_checking is False
        assert feed_state.failure_count == initial_failure_count + 1
        assert feed_state.last_check_time == frozen_now # Compare against the frozen time

        expected_base_backoff = initial_backoff * (BACKOFF_FACTOR ** (initial_failure_count + 1))
        expected_jitter = 0.1
        expected_backoff_with_jitter = expected_base_backoff * (1 + expected_jitter)
        expected_final_backoff = min(expected_backoff_with_jitter, MAX_BACKOFF_MINUTES)

        assert feed_state.current_backoff_minutes == pytest.approx(expected_final_backoff)

        expected_next_check = frozen_now + dt.timedelta(minutes=expected_final_backoff)
        assert feed_state.next_check_time == expected_next_check

        mock_logger.error.assert_any_call(f"Error during scheduled feed check for {feed_url}: Fetch failed!", exc_info=False)
        mock_logger.warning.assert_any_call(f"Failed check for {feed_url} (Attempt {feed_state.failure_count}). Backing off. Next check in ~{expected_final_backoff:.1f} min.")

@patch('scheduler.logger')
def test_scheduled_check_feed_job_already_checking(mock_logger, app_state, feed_state):
    """Test that the job skips if the is_checking flag is set."""
    # --- Setup ---
    target_feed_state = app_state.feed_states["http://example.com/rss"]
    target_feed_state.is_checking = True # Simulate it's already running

    # Patch check_feed locally just to ensure it's not called
    with patch('scheduler.check_feed') as mock_check_feed:
        # --- Action ---
        scheduled_check_feed_job(app_state, target_feed_state)

        # --- Assertions ---
        mock_check_feed.assert_not_called()
        mock_logger.warning.assert_called_once_with(f"Check for {target_feed_state.url} already in progress. Skipping.")
        # Ensure state wasn't changed (beyond the initial is_checking=True)
        assert target_feed_state.is_checking is True # Remains true as the job didn't run

# --- Tests for scheduled_summarize_job ---

@patch('scheduler.process_and_summarize')
@patch('scheduler.logger')
def test_scheduled_summarize_job_with_buffer(
    mock_logger, mock_process_summarize, app_state):
    """Test the summarize job when the buffer has items."""
    # --- Setup ---
    # Add items to buffer
    entry1 = {'title': 'Buffered 1', 'link': 'buf1', 'guid': 'bufguid1'}
    entry2 = {'title': 'Buffered 2', 'link': 'buf2', 'id': 'bufid2'}
    buffer_items = [
        (entry1, "Buf Content 1", "http://feedA.com"),
        (entry2, "Buf Content 2", "http://feedB.com"),
    ]
    app_state.articles_buffer.extend(buffer_items)
    initial_buffer_len = len(app_state.articles_buffer)
    initial_ids = app_state.processed_ids.copy()
    expected_final_ids = initial_ids.union({"final_id"}) # Simulate process_and_summarize returning updated IDs

    # Mock process_and_summarize return value
    mock_process_summarize.return_value = expected_final_ids

    llm_mock = Mock()
    model_name = "summary-model"
    feed_args = {"output_feed_file": "summary.xml", "output_feed_title": "Summary"}
    system_prompt_template = "Test System Prompt Template"

    # --- Action ---
    scheduled_summarize_job(
        app_state,
        llm_mock,
        model_name,
        feed_args,
        system_prompt_template
    )

    # --- Assertions ---
    # Check that process_and_summarize was called correctly
    mock_process_summarize.assert_called_once()
    # Use call_args_list to handle positional or keyword args
    call_args_list = mock_process_summarize.call_args_list
    assert len(call_args_list) == 1
    call_obj = call_args_list[0]

    # Check core arguments (might be pos or kw)
    if call_obj.args:
        assert call_obj.args[0] == buffer_items # articles_to_process
        assert call_obj.args[1] == initial_ids # processed_ids
        assert call_obj.args[2] == llm_mock # llm
        assert call_obj.args[3] == model_name # model_name
    else:
        assert call_obj.kwargs['articles_to_process'] == buffer_items
        assert call_obj.kwargs['processed_ids'] == initial_ids
        assert call_obj.kwargs['llm'] == llm_mock
        assert call_obj.kwargs['model_name'] == model_name

    # Check keyword args passed via **feed_args (these should always be kwargs)
    assert call_obj.kwargs['output_feed_file'] == feed_args['output_feed_file']
    assert call_obj.kwargs['output_feed_title'] == feed_args['output_feed_title']

    # Check AppState updates
    assert len(app_state.articles_buffer) == 0 # Buffer should be cleared
    assert app_state.processed_ids == expected_final_ids # Master IDs updated

    mock_logger.debug.assert_any_call(f"Summarization complete. Processed IDs: {len(expected_final_ids)}")

@patch('scheduler.process_and_summarize')
@patch('scheduler.logger')
def test_scheduled_summarize_job_empty_buffer(
    mock_logger, mock_process_summarize, app_state):
    """Test the summarize job when the buffer is empty."""
    # --- Setup ---
    app_state.articles_buffer.clear()
    initial_ids = app_state.processed_ids.copy()
    llm_mock = Mock()
    model_name = "summary-model"
    feed_args = {"output_feed_file": "summary.xml"}
    system_prompt_template = "Test System Prompt Template"

    # --- Action ---
    scheduled_summarize_job(
        app_state,
        llm_mock,
        model_name,
        feed_args,
        system_prompt_template
    )

    # --- Assertions ---
    mock_process_summarize.assert_not_called() # Should not be called if buffer is empty
    assert app_state.articles_buffer == [] # Still empty
    assert app_state.processed_ids == initial_ids # IDs unchanged
    mock_logger.info.assert_called_once_with("Scheduled summary run: No articles in buffer to process.")

@patch('scheduler.process_and_summarize')
@patch('scheduler.logger')
def test_scheduled_summarize_job_process_exception(
    mock_logger, mock_process_summarize, app_state):
    """Test the summarize job handles exceptions during processing."""
    # --- Setup ---
    entry1 = {'title': 'Buffered 1', 'link': 'buf1', 'guid': 'bufguid1'}
    buffer_items = [(entry1, "Buf Content 1", "http://feedA.com")]
    app_state.articles_buffer.extend(buffer_items)
    initial_ids = app_state.processed_ids.copy()
    llm_mock = Mock()
    model_name = "summary-model"
    feed_args = {"output_feed_file": "summary.xml"}
    system_prompt_template = "Test System Prompt Template"

    # Mock process_and_summarize to raise an exception
    error_message = "Summarization failed badly!"
    mock_process_summarize.side_effect = Exception(error_message)

    # --- Action ---
    scheduled_summarize_job(
        app_state,
        llm_mock,
        model_name,
        feed_args,
        system_prompt_template
    )

    # --- Assertions ---
    mock_process_summarize.assert_called_once() # It was called
    assert app_state.articles_buffer == [] # Buffer is cleared even on exception
    assert app_state.processed_ids == initial_ids # IDs remain unchanged as process_and_summarize failed
    mock_logger.error.assert_called_once_with(
        f"Error during scheduled summarization processing: {error_message}", exc_info=True
    ) 