import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .calculator import calculate_total_time, check_player_compatibility


def planner_index(request):
    """플래너 페이지"""
    return render(request, "planner/index.html")


@method_decorator(csrf_exempt, name="dispatch")
class PlannerCalculateView(View):
    """
    플레이타임 계산 API
    POST /planner/calculate/
    body: {
        "players": 4,
        "games": [
            {"name": "카탄", "play_time": 90, "players": "3~4"},
            {"name": "웬디", "play_time": 120, "players": "4~5"}
        ]
    }
    """

    def post(self, request):
        try:
            body    = json.loads(request.body)
            games   = body.get("games", [])
            players = int(body.get("players", 0))
        except (json.JSONDecodeError, ValueError):
            return JsonResponse({"error": "잘못된 요청 형식입니다."}, status=400)

        if not games:
            return JsonResponse({"error": "게임을 선택해주세요."}, status=400)

        time_result    = calculate_total_time(games)
        compat_result  = check_player_compatibility(games, players) if players else []

        return JsonResponse({
            "time":          time_result,
            "compatibility": compat_result,
        })