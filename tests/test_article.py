import pytest
from unittest.mock import patch, MagicMock
from article import clean_text, fetch_article_content
import requests

def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("test . test , test ? test !") == "test. test, test? test!"
    assert clean_text(None) == None
    assert clean_text("") == ""

@patch('article.requests.get')
def test_fetch_article_content_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'<html><body><article><p>Test content.</p></article></body></html>'
    mock_get.return_value = mock_response

    content = fetch_article_content('http://example.com')
    assert content == 'Test content.'
    mock_get.assert_called_once_with('http://example.com', timeout=15, headers={'User-Agent': 'RSSSummarizerBot/1.0 (+https://github.com/your-repo/rss-summarizer)'})

@patch('article.requests.get')
def test_fetch_article_content_request_exception(mock_get):
    mock_get.side_effect = requests.exceptions.RequestException('Test error')

    content = fetch_article_content('http://example.com')
    assert content is None

@patch('article.requests.get')
def test_fetch_article_content_http_error(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError('Not Found')
    mock_get.return_value = mock_response

    content = fetch_article_content('http://example.com')
    assert content is None

@patch('article.requests.get')
def test_fetch_article_content_no_article_tag(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'<html><body><div><p>Test content.</p></div></body></html>'
    mock_get.return_value = mock_response

    content = fetch_article_content('http://example.com')
    assert content == 'Test content.'

@patch('article.requests.get')
def test_fetch_article_content_no_text(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'<html><body><article></article></body></html>'
    mock_get.return_value = mock_response

    content = fetch_article_content('http://example.com')
    assert content is None
