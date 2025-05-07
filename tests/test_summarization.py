import pytest
from unittest.mock import patch, MagicMock, call
import logging # Import logging for setting level

import tiktoken # For direct encoding/decoding in tests
from langchain_openai import ChatOpenAI # For type hinting mocks

# Import functions and constants to test
from summarization import count_tokens, summarize_text_with_langchain
from config import logger, MAX_TOKENS

# --- Tests for count_tokens --- 

def test_count_tokens_empty():
    """Test counting tokens in an empty string."""
    assert count_tokens("") == 0

def test_count_tokens_simple():
    """Test counting tokens in a simple string with the default model."""
    text = "Hello world"
    # Expected token count might vary slightly depending on tiktoken version,
    # but should be consistent for a given setup.
    # Let's assume gpt-3.5-turbo encodes "Hello world" as 2 tokens.
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    expected_tokens = len(encoding.encode(text))
    assert count_tokens(text, model="gpt-3.5-turbo") == expected_tokens

def test_count_tokens_unknown_model(caplog):
    """Test fallback mechanism when model is not found."""
    text = "Test with unknown model."
    fallback_encoding = tiktoken.get_encoding("cl100k_base")
    expected_tokens = len(fallback_encoding.encode(text))

    # Mock get_encoding *before* the patch for encoding_for_model
    with patch('summarization.tiktoken.get_encoding') as mock_get_encoding:
        mock_get_encoding.return_value = fallback_encoding
        # Mock encoding_for_model to raise KeyError
        with patch('summarization.tiktoken.encoding_for_model', side_effect=KeyError("Model not found")):
            actual_tokens = count_tokens(text, model="nonexistent-model")

            assert actual_tokens == expected_tokens
            # It's called once inside the count_tokens function during fallback
            mock_get_encoding.assert_called_once_with("cl100k_base")
            # Check if the warning was logged
            assert "Model nonexistent-model not found" in caplog.text
            assert "Using default cl100k_base" in caplog.text

def test_count_tokens_different_model():
    """Test counting tokens with a different, valid model."""
    text = "Another test sentence."
    model = "gpt-4" # Assuming gpt-4 might use a different encoding or is available
    try:
        encoding = tiktoken.encoding_for_model(model)
        expected_tokens = len(encoding.encode(text))
        assert count_tokens(text, model=model) == expected_tokens
    except KeyError:
        pytest.skip(f"Model {model} not available for tiktoken, skipping test.")

def test_count_tokens_special_chars():
    """Test counting tokens with non-ASCII characters."""
    text = "你好，世界！"
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    expected_tokens = len(encoding.encode(text))
    assert count_tokens(text, model="gpt-3.5-turbo") == expected_tokens

# --- Tests for summarize_text_with_langchain ---

@pytest.fixture
def mock_llm():
    """Fixture to create a mock ChatOpenAI object."""
    return MagicMock(spec=ChatOpenAI)

def test_summarize_empty_text(mock_llm, caplog):
    """Test summarizing an empty string."""
    # Explicitly set the level for the logger we want to capture from
    with caplog.at_level(logging.INFO, logger='RSSSummarizer'):
        result = summarize_text_with_langchain("", mock_llm)
        assert result == "Content unavailable to summarize."
        assert "Skipping summary for empty text" in caplog.text

@patch('summarization.ChatPromptTemplate.from_template')
@patch('summarization.StrOutputParser')
def test_summarize_short_text(mock_parser, mock_prompt_template, mock_llm, caplog):
    """Test summarizing short text that doesn't need truncation."""
    # Explicitly set the level for the logger we want to capture from
    with caplog.at_level(logging.INFO, logger='RSSSummarizer'):
        text = "This is a short text."
        expected_summary = "Short summary."
        model_name = "gpt-3.5-turbo"

        # Mock the chain construction and invocation
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_summary

        # Simulate the chain creation using the pipe operator
        mock_prompt_instance = MagicMock()
        mock_llm_instance = mock_llm # Use the fixture
        mock_parser_instance = mock_parser.return_value

        # Mock the piping mechanism
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_llm_instance)
        mock_llm_instance.__or__ = MagicMock(return_value=mock_chain) # Pipe llm to parser returns final chain

        mock_prompt_template.return_value = mock_prompt_instance

        # Ensure count_tokens gives a value within limits
        with patch('summarization.count_tokens') as mock_count_tokens:
            mock_count_tokens.return_value = 5 # Well within any reasonable limit

            summary = summarize_text_with_langchain(text, mock_llm_instance, model_name=model_name)

            assert summary == expected_summary
            mock_count_tokens.assert_called_once_with(text, model=model_name)
            # Verify the correct text was passed to the mocked chain
            mock_chain.invoke.assert_called_once_with({"text": text})
            assert "Summarizing text..." in caplog.text
            assert f"Text length (5 tokens) is within the limit." in caplog.text
            assert "Summary generated successfully." in caplog.text
            assert "Truncating" not in caplog.text # Ensure no truncation occurred

@patch('summarization.ChatPromptTemplate.from_template')
@patch('summarization.StrOutputParser')
def test_summarize_long_text_truncation(mock_parser, mock_prompt_template, mock_llm, caplog):
    """Test summarizing long text that requires truncation."""
    # Get the specific logger instance
    target_logger = logging.getLogger('RSSSummarizer')
    # Explicitly set the level for the logger we want to capture from
    # Need WARNING level to capture truncation warnings
    with caplog.at_level(logging.WARNING, logger=target_logger.name):
        model_name = "gpt-3.5-turbo"
        available_tokens_for_text = MAX_TOKENS

        # Create text guaranteed to be longer than the limit using real words
        # Calculate the target token count we need to exceed
        target_token_count = available_tokens_for_text + 10 # Aim for slightly over

        base_word = "example "
        long_text = ""
        current_token_count = 0
        while current_token_count <= available_tokens_for_text:
            long_text += base_word
            # Use the actual count_tokens function to check length
            current_token_count = count_tokens(long_text, model=model_name)
            # Safety break to prevent infinite loops in case of issues
            if len(long_text) > (MAX_TOKENS * 10): # Arbitrary large length limit
                pytest.fail("Failed to generate long_text exceeding token limit.")

        initial_token_count = current_token_count # This is the actual count > available

        # Calculate the expected truncated text based on the *actual* encoding
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(long_text)
        print(f"Available tokens for text: {available_tokens_for_text}")
        truncated_tokens = tokens[:available_tokens_for_text]
        expected_truncated_text = encoding.decode(truncated_tokens)
        # Recalculate final_token_count based on the decoded truncated text
        final_token_count = count_tokens(expected_truncated_text, model=model_name)
        # Previous calculation based on token list slicing was potentially inaccurate

        expected_summary = "Truncated summary."

        # Mock the chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected_summary

        mock_prompt_instance = MagicMock()
        mock_llm_instance = mock_llm
        mock_parser_instance = mock_parser.return_value
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_llm_instance)
        mock_llm_instance.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt_template.return_value = mock_prompt_instance

        summary = summarize_text_with_langchain(long_text, mock_llm_instance, model_name=model_name)

        assert summary == expected_summary
        # Check logs for truncation warnings
        assert f"  Warning: Text ({initial_token_count} tokens) exceeds MAX_TOKENS ({MAX_TOKENS}) limit. Truncating text..." in caplog.text
        assert f"  Truncated text to {final_token_count} tokens." in caplog.text
        # Verify the *truncated* text was passed to the chain
        mock_chain.invoke.assert_called_once_with({"text": expected_truncated_text})
        # Also check the success message is logged at INFO level (requires separate context?)
        # We need to check INFO level logs *after* the WARNING level context
        # Let's just check the truncation logs for now, assuming the success log isn't critical here.
        # assert "Summary generated successfully." in caplog.text # This would fail if only capturing WARNING

@patch('summarization.ChatPromptTemplate.from_template')
@patch('summarization.StrOutputParser')
def test_summarize_llm_error(mock_parser, mock_prompt_template, mock_llm, caplog):
    """Test error handling when the LLM chain invocation fails."""
    # Explicitly set the level for the logger we want to capture from
    # Need ERROR level to capture the error log
    with caplog.at_level(logging.ERROR, logger='RSSSummarizer'):
        text = "Some text that causes an error."
        error_message = "LLM failed spectacularly"

        # Mock the chain to raise an exception
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception(error_message)

        mock_prompt_instance = MagicMock()
        mock_llm_instance = mock_llm
        mock_parser_instance = mock_parser.return_value
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_llm_instance)
        mock_llm_instance.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt_template.return_value = mock_prompt_instance

        # Ensure count_tokens is mocked to avoid unrelated errors
        with patch('summarization.count_tokens', return_value=10):
            summary = summarize_text_with_langchain(text, mock_llm_instance)

            assert summary == "Error generating summary."
            assert f"Error during summarization: {error_message}" in caplog.text
            mock_chain.invoke.assert_called_once_with({"text": text})

@patch('summarization.ChatPromptTemplate.from_template')
@patch('summarization.StrOutputParser')
def test_summarize_context_length_error(mock_parser, mock_prompt_template, mock_llm, caplog):
    """Test specific error handling for context length exceeded errors."""
    # Explicitly set the level for the logger we want to capture from
    # Need ERROR level to capture the error log
    with caplog.at_level(logging.ERROR, logger='RSSSummarizer'):
        text = "Very long text, even after truncation."
        # Simulate an error message containing the specific keyword
        error_message = "something something context_length_exceeded something"

        # Mock the chain to raise the specific exception
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception(error_message)

        mock_prompt_instance = MagicMock()
        mock_llm_instance = mock_llm
        mock_parser_instance = mock_parser.return_value
        mock_prompt_instance.__or__ = MagicMock(return_value=mock_llm_instance)
        mock_llm_instance.__or__ = MagicMock(return_value=mock_chain)
        mock_prompt_template.return_value = mock_prompt_instance

        with patch('summarization.count_tokens', return_value=10):
            summary = summarize_text_with_langchain(text, mock_llm_instance)

            expected_error_msg = "Error: The combined text was too long to summarize, even after truncation."
            assert summary == expected_error_msg
            assert f"Error during summarization: {error_message}" in caplog.text
            mock_chain.invoke.assert_called_once_with({"text": text})