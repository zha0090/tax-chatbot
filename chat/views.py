import logging

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from retrieval.pipeline import get_pipeline

logger = logging.getLogger(__name__)


class ChatView(APIView):
    """Main chat endpoint: routes query through hybrid retrieval + LLM generation."""

    def post(self, request):
        query = request.data.get("query", "").strip()
        if not query:
            return Response(
                {"error": "query is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            pipeline = get_pipeline()
            result = pipeline.answer(query)
            return Response(
                {
                    "query": query,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "routing": result.get("routing_info"),
                }
            )
        except Exception as e:
            logger.exception("Chat pipeline error")
            return Response(
                {"error": f"Internal error: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@api_view(["GET"])
def health_check(request):
    return Response({"status": "ok"})
