from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.core.paginator import Paginator

# TODO: from contents.models.crimescene import CrimeScene


class CrimeSceneListView(View):
    """GET /contents/crimescene/"""

    def get(self, request):
        dummy = [
            {"id": 1, "name": "웬디, 어른이 되렴", "rating": 4.5, "players": "4~5명", "play_time": "120분"},
            {"id": 2, "name": "구두룡 저택의 살인", "rating": 3.3, "players": "7~9명", "play_time": "120분"},
        ]
        paginator = Paginator(dummy, 12)
        page      = paginator.get_page(int(request.GET.get("page", 1)))

        if request.headers.get("Accept") == "application/json":
            return JsonResponse({"results": list(page), "total_pages": paginator.num_pages})

        return render(request, "contents/crimescene/list.html", {"games": page})


class CrimeSceneDetailView(View):
    """GET /contents/crimescene/<int:pk>/"""

    def get(self, request, pk):
        dummy = {
            "id": pk, "name": f"머더미스터리 {pk}", "rating": 4.0,
            "players": "4~5명", "play_time": "120분",
            "description": "설명 placeholder",
            "tags": {"시리즈": "", "제작": "", "출판사": "", "국내 출판사": ""},
            "reviews": [],
        }
        return render(request, "contents/crimescene/detail.html", {"game": dummy})