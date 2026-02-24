from __future__ import annotations

import logging
import time

from django.conf import settings
from openai import APIConnectionError, APIError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    """Create an OpenAI client. Prefer reusing via ChatPipeline.openai_client."""
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_texts(
    texts: list[str],
    model: str | None = None,
    batch_size: int | None = None,
) -> list[list[float]]:
    """Embed a list of texts using the OpenAI embeddings API.

    Handles batching and retries automatically. Returns one embedding
    vector per input text, in the same order.
    """
    if not texts:
        return []

    model = model or settings.OPENAI_EMBEDDING_MODEL
    batch_size = batch_size or settings.OPENAI_EMBED_BATCH_SIZE
    max_chars = settings.OPENAI_EMBED_MAX_CHARS
    client = get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = [t[:max_chars] for t in texts[i : i + batch_size]]

        embeddings = _embed_batch_with_retry(client, batch, model)
        all_embeddings.extend(embeddings)

        if i + batch_size < len(texts):
            logger.info(
                "Embedded %d / %d texts", i + batch_size, len(texts)
            )

    return all_embeddings


def embed_single(text: str, model: str | None = None) -> list[float]:
    """Embed a single text string. Convenience wrapper around embed_texts."""
    return embed_texts([text], model=model)[0]


def _embed_batch_with_retry(
    client: OpenAI,
    batch: list[str],
    model: str,
) -> list[list[float]]:
    max_retries = settings.OPENAI_EMBED_MAX_RETRIES
    for attempt in range(1, max_retries + 1):
        try:
            response = client.embeddings.create(input=batch, model=model)
            return [item.embedding for item in response.data]
        except RateLimitError:
            if attempt == max_retries:
                raise
            wait = 2**attempt
            logger.warning("Rate limited, retrying in %ds (attempt %d)", wait, attempt)
            time.sleep(wait)
        except (APIError, APIConnectionError):
            if attempt == max_retries:
                raise
            wait = 2**attempt
            logger.warning("API error, retrying in %ds (attempt %d)", wait, attempt)
            time.sleep(wait)
