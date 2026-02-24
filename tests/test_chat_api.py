"""Integration tests for the /api/chat/ and /api/health/ endpoints."""

from __future__ import annotations

import os

import pytest
from django.test import Client


def _has_api_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key and key != "sk-your-key-here")


@pytest.fixture()
def api_client():
    return Client()


class TestHealthEndpoint:
    def test_health_check(self, api_client):
        response = api_client.get("/api/health/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestChatEndpoint:
    def test_empty_query_returns_400(self, api_client):
        response = api_client.post(
            "/api/chat/",
            data={"query": ""},
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "error" in response.json()

    def test_missing_query_returns_400(self, api_client):
        response = api_client.post(
            "/api/chat/",
            data={},
            content_type="application/json",
        )
        assert response.status_code == 400

    @pytest.mark.skipif(not _has_api_key(), reason="OPENAI_API_KEY not set")
    def test_valid_query_returns_answer(self, api_client):
        response = api_client.post(
            "/api/chat/",
            data={"query": "What is the average tax rate for corporations?"},
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "routing" in data
        assert len(data["answer"]) > 10

    @pytest.mark.skipif(not _has_api_key(), reason="OPENAI_API_KEY not set")
    def test_response_includes_routing_info(self, api_client):
        response = api_client.post(
            "/api/chat/",
            data={"query": "What is the tax rate in California?"},
            content_type="application/json",
        )
        data = response.json()
        assert "routing" in data
        assert "lanes" in data["routing"]
        assert isinstance(data["routing"]["lanes"], list)

    def test_get_method_not_allowed(self, api_client):
        response = api_client.get("/api/chat/")
        assert response.status_code == 405

    def test_homepage_returns_200(self, api_client):
        response = api_client.get("/")
        assert response.status_code == 200
