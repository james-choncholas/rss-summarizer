import tiktoken
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI # Keep specific import if only using this

# Import logger and constants from config
from config import logger, MAX_TOKENS

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count the number of tokens in a text string."""
    if not text: return 0 # Handle empty text
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        logger.warning(f"Model {model} not found for token counting. Using default cl100k_base.")
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def summarize_text_with_langchain(text_to_summarize, llm: ChatOpenAI, model_name="gpt-3.5-turbo"):
    """Summarizes the given text using LangChain."""
    if not text_to_summarize:
         logger.info("  Skipping summary for empty text.")
         return "Content unavailable to summarize."

    logger.info("  Summarizing text...")

    # Check token count BEFORE creating prompt/chain.
    text_tokens = count_tokens(text_to_summarize, model=model_name)

    if text_tokens > MAX_TOKENS:
        logger.warning(f"  Warning: Text ({text_tokens} tokens) exceeds MAX_TOKENS ({MAX_TOKENS}) limit. Truncating text...")
        # Truncate based on token count directly for better accuracy
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base") # Fallback encoding

        tokens = encoding.encode(text_to_summarize)
        truncated_tokens = tokens[:MAX_TOKENS]
        text_to_summarize = encoding.decode(truncated_tokens)
        # Log the new token count after truncation
        final_token_count = count_tokens(text_to_summarize, model=model_name)
        logger.warning(f"  Truncated text to {final_token_count} tokens.")
    else:
        logger.info(f"  Text length ({text_tokens} tokens) is within the limit.")


    # Use a slightly more robust prompt
    prompt_template = """{text}

    CONCISE SUMMARY:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | llm | StrOutputParser()

    try:
        summary = chain.invoke({"text": text_to_summarize})
        logger.info("  Summary generated successfully.")
        # Basic post-processing: remove potential leading/trailing whitespace
        return summary.strip()
    except Exception as e:
        logger.error(f"  Error during summarization: {e}", exc_info=True)
        # Consider more specific error handling (e.g., context length exceeded)
        if "context_length_exceeded" in str(e).lower():
             return "Error: The combined text was too long to summarize, even after truncation."
        return "Error generating summary." 