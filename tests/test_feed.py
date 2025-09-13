import pytest
from unittest.mock import patch, MagicMock
from feed import fetch_rss_feed, check_feed

@patch('feed.feedparser.parse')
def test_fetch_rss_feed_success(mock_parse):
    mock_feed = MagicMock()
    mock_feed.bozo = False
    mock_feed.entries = [{'title': 'Test Entry'}]
    mock_parse.return_value = mock_feed

    entries = fetch_rss_feed('http://example.com/rss')
    assert entries == [{'title': 'Test Entry'}]

@patch('feed.feedparser.parse')
def test_fetch_rss_feed_bozo_exception(mock_parse):
    mock_feed = MagicMock()
    mock_feed.bozo = True
    mock_feed.bozo_exception = 'Test Exception'
    mock_parse.return_value = mock_feed

    with pytest.raises(Exception, match='Test Exception'):
        fetch_rss_feed('http://example.com/rss')

@patch('feed.feedparser.parse')
def test_fetch_rss_feed_no_entries(mock_parse):
    mock_feed = MagicMock()
    mock_feed.bozo = False
    mock_feed.entries = []
    mock_parse.return_value = mock_feed

    with pytest.raises(Exception, match='No entries found in the feed.'):
        fetch_rss_feed('http://example.com/rss')

@patch('feed.fetch_rss_feed')
@patch('feed.fetch_article_content')
def test_check_feed_new_articles(mock_fetch_content, mock_fetch_rss):
    mock_fetch_rss.return_value = [
        {'guid': '1', 'title': 'New Article', 'link': 'http://example.com/new'},
        {'guid': '2', 'title': 'Old Article', 'link': 'http://example.com/old'}
    ]
    mock_fetch_content.return_value = 'Article content'
    processed_ids = {'2'}

    new_entries, updated_ids = check_feed('http://example.com/rss', False, processed_ids)

    assert len(new_entries) == 1
    assert new_entries[0][0]['title'] == 'New Article'
    assert new_entries[0][1] == 'Article content'
    assert updated_ids == {'1', '2'}

@patch('feed.fetch_rss_feed')
def test_check_feed_use_feed_summary(mock_fetch_rss):
    mock_fetch_rss.return_value = [
        {'guid': '1', 'title': 'New Article', 'summary': 'Article summary'}
    ]
    processed_ids = set()

    new_entries, updated_ids = check_feed('http://example.com/rss', True, processed_ids)

    assert len(new_entries) == 1
    assert new_entries[0][1] == 'Article summary'
    assert updated_ids == {'1'}

@patch('feed.fetch_rss_feed')
def test_check_feed_no_new_articles(mock_fetch_rss):
    mock_fetch_rss.return_value = [
        {'guid': '1', 'title': 'Old Article'}
    ]
    processed_ids = {'1'}

    new_entries, updated_ids = check_feed('http://example.com/rss', False, processed_ids)

    assert len(new_entries) == 0
    assert updated_ids == {'1'}
