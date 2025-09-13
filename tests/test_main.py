import pytest
from unittest.mock import patch, MagicMock
import main
from scheduler import AppState

@patch('main.argparse.ArgumentParser')
def test_main_run_once(mock_argparse):
    mock_args = MagicMock()
    mock_args.run_once = True
    mock_args.feed_urls = ['http://example.com/rss']
    mock_args.summary_time = '08:00'
    mock_args.check_interval = 30
    mock_args.use_feed_summary = False
    mock_args.api_url = None
    mock_args.api_model = 'gpt-4o-mini'
    mock_args.temperature = 0.3
    mock_args.system_prompt = 'Test prompt'
    mock_args.output_feed_file = 'output/summary_feed.xml'
    mock_args.output_feed_title = 'Daily Summarized Feed'
    mock_args.output_feed_link = 'http://localhost:8000/output/summary_feed.xml'
    mock_args.output_feed_description = 'Summaries of articles from monitored feeds.'
    mock_args.port = 8000
    mock_args.serve_dir = '.'
    mock_argparse.return_value.parse_args.return_value = mock_args

    # This function will be called by the test to simulate scheduled_check_feed_job
    def check_feed_job_side_effect(app_state, feed_url, use_feed_summary):
        app_state.articles_buffer.append(({'title': 'Test Article'}, 'Test Content', feed_url))

    with patch('main.run_http_server'), \
         patch('main.load_processed_ids', return_value=set()), \
         patch('main.scheduled_check_feed_job', side_effect=check_feed_job_side_effect), \
         patch('main.process_and_summarize') as mock_process, \
         patch('main.time.sleep'):
        main.main()
        mock_process.assert_called()

@patch('main.argparse.ArgumentParser')
def test_main_scheduler(mock_argparse):
    mock_args = MagicMock()
    mock_args.run_once = False
    mock_args.feed_urls = ['http://example.com/rss']
    mock_args.summary_time = '08:00'
    mock_args.check_interval = 30
    # Set other necessary args to default values
    mock_args.use_feed_summary = False
    mock_args.api_url = None
    mock_args.api_model = 'gpt-4o-mini'
    mock_args.temperature = 0.3
    mock_args.system_prompt = 'Test prompt'
    mock_args.output_feed_file = 'output/summary_feed.xml'
    mock_args.output_feed_title = 'Daily Summarized Feed'
    mock_args.output_feed_link = 'http://localhost:8000/output/summary_feed.xml'
    mock_args.output_feed_description = 'Summaries of articles from monitored feeds.'
    mock_args.port = 8000
    mock_args.serve_dir = '.'
    mock_argparse.return_value.parse_args.return_value = mock_args

    with patch('main.run_http_server'), \
         patch('main.load_processed_ids', return_value=set()), \
         patch('main.run_scheduler') as mock_run_scheduler, \
         patch('main.time.sleep', side_effect=KeyboardInterrupt): # to exit the loop
        main.main()
        mock_run_scheduler.assert_called()