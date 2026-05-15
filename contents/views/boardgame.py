# contents/views/boardgame.py

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views import View
from django.core.paginator import Paginator

# TODO: 전처리 완료 후 실제 모델로 교체
# from contents.models.boardgame import BoardGame


class BoardGameListView(View):
    """
    GET /contents/boardgame/
    params: page, players, max_time, q
    """

    def get(self, request):
        # TODO: 실제 DB 쿼리로 교체
        # qs = BoardGame.objects.all()
        # if p := request.GET.get("players"):  qs = qs.filter(...)
        # if t := request.GET.get("max_time"): qs = qs.filter(play_time__lte=t)
        # if q := request.GET.get("q"):        qs = qs.filter(name__icontains=q)

        dummy = [
            {"id": 1, "name": "카탄", "rating": 4.2, "players": "3~4명", "play_time": "90분", "category": "boardgame"},
            {"id": 2, "name": "아줄",  "rating": 4.5, "players": "2~4명", "play_time": "45분", "category": "boardgame"},
        ]
        paginator = Paginator(dummy, 12)
        page      = paginator.get_page(int(request.GET.get("page", 1)))

        if request.headers.get("Accept") == "application/json":
            return JsonResponse({"results": list(page), "total_pages": paginator.num_pages})

        return render(request, "contents/boardgame/list.html", {"games": page})


class BoardGameDetailView(View):
    """GET /contents/boardgame/<int:pk>/"""

    def get(self, request, pk):
        # TODO: 실제 DB 조회로 교체
        # game = get_object_or_404(BoardGame, pk=pk)
        dummy = {
            "id": pk, "name": f"보드게임 {pk}", "rating": 4.2,
            "players": "2~4명", "play_time": "60분",
            "description": "설명 placeholder", "tags": {}, "reviews": [],
        }
        return render(request, "contents/boardgame/detail.html", {"game": dummy})