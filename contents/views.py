import json
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views import View
from django.core.paginator import Paginator

# TODO: 전처리 완료 후 실제 모델로 교체
# from contents.models.boardgame import BoardGame
# from contents.models.escape import Escape
# from contents.models.crimescene import CrimeScene


# ── 더미 데이터 (전처리 완료 전 구조 확인용) ──────────────────
DUMMY_BOARDGAMES = [
    {"id": 1, "name": "카탄",       "rating": 4.2, "players": "3~4명", "play_time": "90분",  "difficulty": "중급", "tags": ["전략", "협력"]},
    {"id": 2, "name": "아줄",       "rating": 4.5, "players": "2~4명", "play_time": "45분",  "difficulty": "초급", "tags": ["타일", "추상"]},
    {"id": 3, "name": "윙스팬",     "rating": 4.6, "players": "1~5명", "play_time": "60분",  "difficulty": "중급", "tags": ["엔진빌딩", "자연"]},
    {"id": 4, "name": "팬데믹",     "rating": 4.3, "players": "2~4명", "play_time": "60분",  "difficulty": "중급", "tags": ["협력", "전략"]},
]

DUMMY_ESCAPES = [
    {"id": 1, "name": "비밀의 방",  "rating": 4.3, "players": "2~5명", "play_time": "70분",  "difficulty": "중",   "region": "홍대",  "tags": ["추리", "초보 추천"]},
    {"id": 2, "name": "저주받은 저택", "rating": 4.1, "players": "2~4명", "play_time": "60분", "difficulty": "상",  "region": "강남",  "tags": ["공포", "어드벤처"]},
]

DUMMY_CRIMESCENES = [
    {"id": 1, "name": "웬디, 어른이 되렴", "rating": 4.5, "players": "4~5명", "play_time": "120분", "difficulty": "중급", "tags": ["추리", "입문용"]},
    {"id": 2, "name": "구두룡 저택의 살인", "rating": 3.3, "players": "7~9명", "play_time": "120분", "difficulty": "고급", "tags": ["추리", "협력"]},
]


# =====================================================================
# 보드게임
# =====================================================================

class BoardGameListView(View):
    """
    GET /contents/boardgame/
    params: page, q, difficulty
    """

    def get(self, request):
        games = DUMMY_BOARDGAMES  # TODO: BoardGame.objects.all()

        # 검색 필터
        q = request.GET.get("q", "")
        if q:
            games = [g for g in games if q in g["name"]]

        # 난이도 필터
        difficulty = request.GET.get("difficulty", "")
        if difficulty:
            games = [g for g in games if g["difficulty"] == difficulty]

        paginator = Paginator(games, 12)
        page      = paginator.get_page(int(request.GET.get("page", 1)))

        if request.headers.get("Accept") == "application/json":
            return JsonResponse({
                "results":     list(page.object_list),
                "total_pages": paginator.num_pages,
                "count":       paginator.count,
            })

        return render(request, "contents/boardgame/list.html", {
            "games":      page,
            "q":          q,
            "difficulty": difficulty,
        })


class BoardGameDetailView(View):
    """GET /contents/boardgame/<int:pk>/"""

    def get(self, request, pk):
        # TODO: game = get_object_or_404(BoardGame, pk=pk)
        game = next((g for g in DUMMY_BOARDGAMES if g["id"] == pk), None)
        if not game:
            return JsonResponse({"error": "게임을 찾을 수 없습니다."}, status=404)

        if request.headers.get("Accept") == "application/json":
            return JsonResponse(game)

        return render(request, "contents/boardgame/detail.html", {"game": game})


# =====================================================================
# 방탈출
# =====================================================================

class EscapeListView(View):
    """
    GET /contents/escape/
    params: page, q, region, difficulty
    """

    def get(self, request):
        games = DUMMY_ESCAPES  # TODO: Escape.objects.all()

        q = request.GET.get("q", "")
        if q:
            games = [g for g in games if q in g["name"]]

        region = request.GET.get("region", "")
        if region:
            games = [g for g in games if g["region"] == region]

        difficulty = request.GET.get("difficulty", "")
        if difficulty:
            games = [g for g in games if g["difficulty"] == difficulty]

        paginator = Paginator(games, 12)
        page      = paginator.get_page(int(request.GET.get("page", 1)))

        if request.headers.get("Accept") == "application/json":
            return JsonResponse({
                "results":     list(page.object_list),
                "total_pages": paginator.num_pages,
                "count":       paginator.count,
            })

        return render(request, "contents/escape/list.html", {
            "games":      page,
            "q":          q,
            "region":     region,
            "difficulty": difficulty,
        })


class EscapeDetailView(View):
    """GET /contents/escape/<int:pk>/"""

    def get(self, request, pk):
        game = next((g for g in DUMMY_ESCAPES if g["id"] == pk), None)
        if not game:
            return JsonResponse({"error": "게임을 찾을 수 없습니다."}, status=404)

        if request.headers.get("Accept") == "application/json":
            return JsonResponse(game)

        return render(request, "contents/escape/detail.html", {"game": game})


# =====================================================================
# 머더미스터리 (크라임씬)
# =====================================================================

class CrimeSceneListView(View):
    """
    GET /contents/crimescene/
    params: page, q, difficulty
    """

    def get(self, request):
        games = DUMMY_CRIMESCENES  # TODO: CrimeScene.objects.all()

        q = request.GET.get("q", "")
        if q:
            games = [g for g in games if q in g["name"]]

        difficulty = request.GET.get("difficulty", "")
        if difficulty:
            games = [g for g in games if g["difficulty"] == difficulty]

        paginator = Paginator(games, 12)
        page      = paginator.get_page(int(request.GET.get("page", 1)))

        if request.headers.get("Accept") == "application/json":
            return JsonResponse({
                "results":     list(page.object_list),
                "total_pages": paginator.num_pages,
                "count":       paginator.count,
            })

        return render(request, "contents/crimescene/list.html", {
            "games":      page,
            "q":          q,
            "difficulty": difficulty,
        })


class CrimeSceneDetailView(View):
    """GET /contents/crimescene/<int:pk>/"""

    def get(self, request, pk):
        game = next((g for g in DUMMY_CRIMESCENES if g["id"] == pk), None)
        if not game:
            return JsonResponse({"error": "게임을 찾을 수 없습니다."}, status=404)

        if request.headers.get("Accept") == "application/json":
            return JsonResponse(game)

        return render(request, "contents/crimescene/detail.html", {"game": game})