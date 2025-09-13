import pytest
from unittest.mock import patch, MagicMock, call
import logging
import tiktoken
from langchain_openai import ChatOpenAI

from summarization import count_tokens, summarize_text_with_langchain
from config import logger, MAX_TOKENS

# --- Tests for count_tokens --- 

def test_count_tokens_empty():
    assert count_tokens("") == 0

def test_count_tokens_simple():
    text = "Hello world"
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    expected_tokens = len(encoding.encode(text))
    assert count_tokens(text, model="gpt-3.5-turbo") == expected_tokens

def test_count_tokens_unknown_model(caplog):
    text = "Test with unknown model."
    fallback_encoding = tiktoken.get_encoding("cl100k_base")
    expected_tokens = len(fallback_encoding.encode(text))

    with patch('summarization.tiktoken.encoding_for_model', side_effect=KeyError("Model not found")):
        actual_tokens = count_tokens(text, model="nonexistent-model")
        assert actual_tokens == expected_tokens
        assert "Model nonexistent-model not found" in caplog.text

def test_count_tokens_special_chars():
    text = "你好，世界！"
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    expected_tokens = len(encoding.encode(text))
    assert count_tokens(text, model="gpt-3.5-turbo") == expected_tokens

# --- Tests for summarize_text_with_langchain ---

@pytest.fixture
def mock_llm():
    return MagicMock(spec=ChatOpenAI)

def test_summarize_empty_text(mock_llm, caplog):
    with caplog.at_level(logging.INFO, logger='RSSSummarizer'):
        result = summarize_text_with_langchain("", mock_llm)
        assert result == "Content unavailable to summarize."
        assert "Skipping summary for empty text" in caplog.text

@patch('summarization.ChatPromptTemplate.from_template')
@patch('summarization.StrOutputParser')
def test_summarize_short_text(mock_parser, mock_prompt_template, mock_llm, caplog):
    with caplog.at_level(logging.INFO, logger='RSSSummarizer'):
        text = "This is a short text."
        expected_summary = "Short summary."
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_summary
        mock_prompt_instance = MagicMock()
        mock_llm.__or__.return_value = mock_chain
        mock_prompt_template.return_value = mock_prompt_instance
        mock_prompt_instance.__or__.return_value = mock_llm

        with patch('summarization.count_tokens', return_value=5):
            summary = summarize_text_with_langchain(text, mock_llm, model_name="gpt-3.5-turbo")
            assert summary == expected_summary
            mock_chain.invoke.assert_called_once_with({"text": text})

@patch('summarization.ChatPromptTemplate.from_template')
@patch('summarization.StrOutputParser')
def test_summarize_long_text_truncation(mock_parser, mock_prompt_template, mock_llm, caplog):
    with caplog.at_level(logging.WARNING, logger='RSSSummarizer'):
        long_text = "word " * (MAX_TOKENS + 1)
        expected_summary = "Truncated summary."
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_summary
        mock_prompt_instance = MagicMock()
        mock_llm.__or__.return_value = mock_chain
        mock_prompt_template.return_value = mock_prompt_instance
        mock_prompt_instance.__or__.return_value = mock_llm

        summary = summarize_text_with_langchain(long_text, mock_llm, model_name="gpt-3.5-turbo")
        assert summary == expected_summary
        assert "exceeds MAX_TOKENS" in caplog.text

@patch('summarization.ChatPromptTemplate.from_template')
@patch('summarization.StrOutputParser')
def test_summarize_llm_error(mock_parser, mock_prompt_template, mock_llm, caplog):
    with caplog.at_level(logging.ERROR, logger='RSSSummarizer'):
        text = "Some text that causes an error."
        error_message = "LLM failed spectacularly"
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception(error_message)
        mock_prompt_instance = MagicMock()
        mock_llm.__or__.return_value = mock_chain
        mock_prompt_template.return_value = mock_prompt_instance
        mock_prompt_instance.__or__.return_value = mock_llm

        with patch('summarization.count_tokens', return_value=10):
            summary = summarize_text_with_langchain(text, mock_llm)
            assert summary == "Error generating summary."
            assert error_message in caplog.text

@patch('summarization.ChatPromptTemplate.from_template')
@patch('summarization.StrOutputParser')
def test_summarize_context_length_error(mock_parser, mock_prompt_template, mock_llm, caplog):
    with caplog.at_level(logging.ERROR, logger='RSSSummarizer'):
        text = "Very long text."
        error_message = "context_length_exceeded"
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception(error_message)
        mock_prompt_instance = MagicMock()
        mock_llm.__or__.return_value = mock_chain
        mock_prompt_template.return_value = mock_prompt_instance
        mock_prompt_instance.__or__.return_value = mock_llm

        with patch('summarization.count_tokens', return_value=10):
            summary = summarize_text_with_langchain(text, mock_llm)
            assert "The combined text was too long to summarize" in summary
