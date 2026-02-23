from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView


class ChatView(APIView):
    """Main chat endpoint that accepts user queries and returns AI-generated answers."""

    def post(self, request):
        query = request.data.get("query", "").strip()
        if not query:
            return Response(
                {"error": "query is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # TODO: Wire up retrieval pipeline + LLM generation
        return Response(
            {
                "query": query,
                "answer": "Pipeline not yet connected.",
                "sources": [],
            }
        )


@api_view(["GET"])
def health_check(request):
    return Response({"status": "ok"})
