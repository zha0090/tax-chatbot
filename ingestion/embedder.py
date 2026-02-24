from __future__ import annotations

import logging
import time

from django.conf import settings
from openai import OpenAI

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
MAX_RETRIES = 3


def get_client() -> OpenAI:
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def embed_texts(
    texts: list[str],
    model: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> list[list[float]]:
    """Embed a list of texts using the OpenAI embeddings API.

    Handles batching and retries automatically. Returns one embedding
    vector per input text, in the same order.
    """
    model = model or settings.OPENAI_EMBEDDING_MODEL
    client = get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch = [t[:8000] for t in batch]

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
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(input=batch, model=model)
            return [item.embedding for item in response.data]
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            wait = 2**attempt
            logger.warning(
                "Embedding attempt %d failed, retrying in %ds...",
                attempt,
                wait,
            )
            time.sleep(wait)
    return []
