from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.core.paginator import Paginator

# TODO: from contents.models.escape import Escape


class EscapeListView(View):
    """GET /contents/escape/"""

    def get(self, request):
        dummy = [
            {"id": 1, "name": "방탈출 샘플 A", "rating": 4.3, "players": "2~6명", "play_time": "60분", "region": "홍대"},
            {"id": 2, "name": "방탈출 샘플 B", "rating": 4.0, "players": "2~4명", "play_time": "60분", "region": "강남"},
        ]
        paginator = Paginator(dummy, 12)
        page      = paginator.get_page(int(request.GET.get("page", 1)))

        if request.headers.get("Accept") == "application/json":
            return JsonResponse({"results": list(page), "total_pages": paginator.num_pages})

        return render(request, "contents/escape/list.html", {"games": page})


class EscapeDetailView(View):
    """GET /contents/escape/<int:pk>/"""

    def get(self, request, pk):
        dummy = {
            "id": pk, "name": f"방탈출 {pk}", "rating": 4.1,
            "players": "2~6명", "play_time": "60분",
            "region": "홍대", "difficulty": "중급",
            "description": "설명 placeholder", "reviews": [],
        }
        return render(request, "contents/escape/detail.html", {"game": dummy})