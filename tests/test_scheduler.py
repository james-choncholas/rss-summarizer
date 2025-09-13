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
    run_scheduler,
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
        state.next_check_time = initial_timestamp
        assert state.next_check_time == initial_timestamp
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
        feed1_state = state.feed_states["http://feed1.com"]
        assert isinstance(feed1_state, FeedState)
        assert feed1_state.next_check_time == initial_timestamp

# --- Tests for process_and_summarize ---

@patch('scheduler.summarize_text_with_langchain')
@patch('scheduler.generate_summary_feed')
@patch('scheduler.save_processed_ids')
@patch('scheduler.logger')
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

    assert updated_ids == initial_ids
    mock_summarize.assert_not_called()
    mock_generate_feed.assert_called_once()
    mock_save_ids.assert_called_once_with(initial_ids)

@patch('scheduler.summarize_text_with_langchain')
@patch('scheduler.generate_summary_feed')
@patch('scheduler.save_processed_ids')
@patch('scheduler.logger')
def test_process_and_summarize_with_articles(
    mock_logger, mock_save_ids, mock_generate_feed, mock_summarize, app_state):
    """Test processing with a list of articles."""
    articles_to_process = [
        ({'title': 'Article 1', 'link': 'http://link1.com', 'guid': 'guid1'}, "Content 1", "http://feedA.com"),
    ]
    initial_ids = app_state.processed_ids.copy()
    mock_summarize.return_value = "This is the combined summary."
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

    assert "guid1" in updated_ids
    mock_summarize.assert_called_once()
    mock_generate_feed.assert_called_once()
    mock_save_ids.assert_called_once()

# --- Tests for scheduled_check_feed_job ---

@patch('scheduler.check_feed')
@patch('scheduler.logger')
@patch('scheduler.random')
def test_scheduled_check_feed_job_success(
    mock_random, mock_logger, mock_check_feed, app_state, initial_timestamp):
    """Test scheduled_check_feed_job on successful feed fetch."""
    feed_url = "http://example.com/rss"
    feed_state = app_state.feed_states[feed_url]
    initial_ids = app_state.processed_ids.copy()
    new_entries_tuples = [({'title': 'New Article 1', 'guid': 'newguid1'}, "New Content 1")]
    updated_ids_from_check = initial_ids.union({"newguid1"})
    mock_check_feed.return_value = (new_entries_tuples, updated_ids_from_check)

    frozen_now = initial_timestamp + dt.timedelta(minutes=1)
    with freeze_time(frozen_now):
        mock_random.uniform.return_value = 0
        scheduled_check_feed_job(app_state, feed_state)
        assert len(app_state.articles_buffer) == 1
        assert app_state.processed_ids == updated_ids_from_check
        assert feed_state.failure_count == 0

@patch('scheduler.check_feed')
@patch('scheduler.logger')
@patch('scheduler.random')
def test_scheduled_check_feed_job_failure_and_backoff(
    mock_random, mock_logger, mock_check_feed, app_state, initial_timestamp):
    """Test scheduled_check_feed_job on failed feed fetch and backoff calculation."""
    feed_url = "http://another.com/feed"
    feed_state = app_state.feed_states[feed_url]
    mock_check_feed.side_effect = Exception("Fetch failed!")

    frozen_now = initial_timestamp + dt.timedelta(minutes=1)
    with freeze_time(frozen_now):
        mock_random.uniform.return_value = 0.1
        scheduled_check_feed_job(app_state, feed_state)
        assert feed_state.failure_count == 1
        assert feed_state.current_backoff_minutes > app_state.initial_check_interval

@patch('scheduler.logger')
def test_scheduled_check_feed_job_already_checking(mock_logger, app_state):
    """Test that the job skips if the is_checking flag is set."""
    target_feed_state = app_state.feed_states["http://example.com/rss"]
    target_feed_state.is_checking = True
    with patch('scheduler.check_feed') as mock_check_feed:
        scheduled_check_feed_job(app_state, target_feed_state)
        mock_check_feed.assert_not_called()

# --- Tests for scheduled_summarize_job ---

@patch('scheduler.process_and_summarize')
@patch('scheduler.logger')
def test_scheduled_summarize_job_with_buffer(
    mock_logger, mock_process_summarize, app_state):
    """Test the summarize job when the buffer has items."""
    app_state.articles_buffer.extend([("entry", "content", "url")])
    expected_final_ids = app_state.processed_ids.union({"final_id"})
    mock_process_summarize.return_value = expected_final_ids
    llm_mock = Mock()

    scheduled_summarize_job(app_state, llm_mock, "model", {}, "prompt")
    mock_process_summarize.assert_called_once()
    assert len(app_state.articles_buffer) == 0
    assert app_state.processed_ids == expected_final_ids

@patch('scheduler.process_and_summarize')
@patch('scheduler.logger')
def test_scheduled_summarize_job_empty_buffer(
    mock_logger, mock_process_summarize, app_state):
    """Test the summarize job when the buffer is empty."""
    app_state.articles_buffer.clear()
    llm_mock = Mock()

    scheduled_summarize_job(app_state, llm_mock, "model", {}, "prompt")
    mock_process_summarize.assert_not_called()

@patch('scheduler.process_and_summarize', side_effect=Exception("Test error"))
@patch('scheduler.logger')
def test_scheduled_summarize_job_exception(
    mock_logger, mock_process_summarize, app_state):
    """Test exception handling in the summarize job."""
    app_state.articles_buffer.extend([("entry", "content", "url")])
    llm_mock = Mock()

    scheduled_summarize_job(app_state, llm_mock, "model", {}, "prompt")
    mock_logger.error.assert_called()

# --- Tests for run_scheduler ---

@patch('scheduler.schedule')
@patch('scheduler.time.sleep', side_effect=KeyboardInterrupt) # To exit the loop
def test_run_scheduler_setup(mock_sleep, mock_schedule, app_state):
    """Test that the scheduler sets up jobs correctly."""
    llm_mock = Mock()
    summary_time = "12:00"
    
    with pytest.raises(KeyboardInterrupt):
        run_scheduler(app_state, 30, summary_time, llm_mock, "model", {}, "prompt")
    
    mock_schedule.every.return_value.day.at.return_value.do.assert_called_once()