"""Query router: classifies user queries and extracts entities.

Uses GPT-4o-mini to determine which retrieval lanes to activate
and extract structured entities (taxpayer type, state, year, etc.)
from natural language questions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from django.conf import settings
from openai import OpenAI

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are a query classifier for a financial/tax chatbot. Given a user question, output JSON with:

1. "lanes": array of retrieval strategies to use. Pick one or more from:
   - "chitchat": for greetings, thanks, or non-tax questions (hi, hello, thanks, what can you do, etc.)
   - "graph": for questions about relationships, comparisons, rankings, or aggregations over structured tax data (taxpayer types, states, years, income sources, deduction types).
   - "vector": for questions about tax rules, IRS form instructions, legal code, or conceptual/definitional questions.
   - "structured": for questions needing exact numerical lookups from the CSV dataset (totals, counts, specific amounts).

2. "entities": object with extracted values (only include if present in the query):
   - "taxpayer_type": one of "Individual", "Corporation", "Partnership", "Non-Profit", "Trust"
   - "state": two-letter state code (CA, NY, TX, FL, PA, IL, OH, GA, NC, MI)
   - "income_source": one of "Salary", "Business Income", "Capital Gains", "Investment", "Rental", "Royalties"
   - "deduction_type": one of "Mortgage Interest", "Business Expenses", "Education Expenses", "Medical Expenses", "Charitable Contributions"
   - "tax_year": integer year (2019-2023)

3. "rewritten_query": a clearer version of the question optimized for search retrieval.

Important: If a question asks about "the highest/lowest/total" across ALL data without specifying a filter, still use "graph" and "structured" lanes with no entities — the system will handle aggregations across the full dataset.

Reply ONLY with valid JSON, no markdown."""

ROUTER_EXAMPLES = [
    {
        "role": "user",
        "content": "What is the average tax rate for corporations in California?",
    },
    {
        "role": "assistant",
        "content": '{"lanes":["graph","structured"],"entities":{"taxpayer_type":"Corporation","state":"CA"},"rewritten_query":"average tax rate for corporations in California"}',
    },
    {
        "role": "user",
        "content": "How do I report rental income on Form 1040?",
    },
    {
        "role": "assistant",
        "content": '{"lanes":["vector"],"entities":{"income_source":"Rental"},"rewritten_query":"how to report rental income on IRS Form 1040"}',
    },
    {
        "role": "user",
        "content": "Which state has the highest tax owed for partnerships in 2022?",
    },
    {
        "role": "assistant",
        "content": '{"lanes":["graph","structured"],"entities":{"taxpayer_type":"Partnership","tax_year":"2022"},"rewritten_query":"state with highest total tax owed for partnerships in 2022"}',
    },
    {
        "role": "user",
        "content": "What deduction type is most common for trusts?",
    },
    {
        "role": "assistant",
        "content": '{"lanes":["graph","structured"],"entities":{"taxpayer_type":"Trust"},"rewritten_query":"most common deduction type for trusts by count"}',
    },
    {
        "role": "user",
        "content": "What is the minimum and maximum tax rate for corporations?",
    },
    {
        "role": "assistant",
        "content": '{"lanes":["graph","structured"],"entities":{"taxpayer_type":"Corporation"},"rewritten_query":"minimum and maximum tax rate for corporations"}',
    },
    {
        "role": "user",
        "content": "How does the US tax rate compare to other countries?",
    },
    {
        "role": "assistant",
        "content": '{"lanes":["vector"],"entities":{},"rewritten_query":"US overall tax rate comparison to other industrialized countries"}',
    },
    {
        "role": "user",
        "content": "hi",
    },
    {
        "role": "assistant",
        "content": '{"lanes":["chitchat"],"entities":{},"rewritten_query":"greeting"}',
    },
    {
        "role": "user",
        "content": "What is the highest tax rate?",
    },
    {
        "role": "assistant",
        "content": '{"lanes":["graph","structured"],"entities":{},"rewritten_query":"highest tax rate across all taxpayers and states"}',
    },
    {
        "role": "user",
        "content": "What can you explain to me?",
    },
    {
        "role": "assistant",
        "content": '{"lanes":["chitchat"],"entities":{},"rewritten_query":"capabilities"}',
    },
    {
        "role": "user",
        "content": "Which state has the lowest tax rate?",
    },
    {
        "role": "assistant",
        "content": '{"lanes":["graph","structured"],"entities":{},"rewritten_query":"state with the lowest average tax rate"}',
    },
]


def classify_query(
    query: str, client: OpenAI | None = None
) -> dict[str, Any]:
    """Classify a user query into retrieval lanes and extract entities.

    Returns dict with keys: lanes, entities, rewritten_query.
    Falls back to vector search if classification fails.
    """
    fallback = {"lanes": ["vector"], "entities": {}, "rewritten_query": query}
    try:
        client = client or OpenAI(api_key=settings.OPENAI_API_KEY)
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            *ROUTER_EXAMPLES,
            {"role": "user", "content": query},
        ]
        response = client.chat.completions.create(
            model=settings.OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=settings.OPENAI_ROUTER_MAX_TOKENS,
        )
        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        result.setdefault("lanes", ["vector"])
        result.setdefault("entities", {})
        result.setdefault("rewritten_query", query)
        if result["entities"].get("tax_year"):
            result["entities"]["tax_year"] = str(result["entities"]["tax_year"])
        return result
    except json.JSONDecodeError:
        logger.warning("Router returned invalid JSON for query: %s", query)
        return fallback
    except Exception:
        logger.warning("Query classification failed", exc_info=True)
        return fallback
