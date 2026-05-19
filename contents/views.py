import json
import re
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.core.paginator import Paginator
from .models import BoardGame, Escape, CrimeScene

CAT_EMOJI = {"escape": "🔐", "boardgame": "🎲", "crimescene": "🕵️"}
CAT_LABEL = {"escape": "방탈출", "boardgame": "보드게임", "crimescene": "크라임씬"}

PAGE_SIZE = 30

DIFFICULTY_MAP = {
    "입문": "초급", "초급": "초급",
    "중":   "중급", "중급": "중급", "중상": "중급",
    "고급": "고급",
    "하":   "초급",
    "상":   "고급",
}

HORROR_KEYWORDS = {"공포", "공포 없음", "공포 약함", "공포 있음", "공포 강함"}

def _filter_tags(tags):
    """공포 관련 태그를 제거한 리스트 반환"""
    if not isinstance(tags, list):
        return []
    return [t for t in tags if t not in HORROR_KEYWORDS]

def _normalize_difficulty(val):
    if not val:
        return ""
    return DIFFICULTY_MAP.get(val.strip(), val.strip())


def _is_korean(name):
    return bool(re.search('[가-힣]', str(name)))


def _normalized_rating(activity):
    """별점을 5점 만점 기준으로 정규화 (정렬 비교용)"""
    rating     = activity.get("rating") or 0
    rating_max = activity.get("rating_max") or 5
    return rating / rating_max * 5


def _sort_key(a):
    """
    정렬 기준:
    1순위: 한국어(boardlife) 게임 우선 (is_korean=True → 10점 가산)
    2순위: 5점 만점 기준 정규화 별점 내림차순
    """
    is_korean_bonus = 10 if _is_korean(a.get("title", "")) else 0
    return is_korean_bonus + _normalized_rating(a)


def _to_activity_boardgame(g):
    is_korean  = _is_korean(g.name)
    source     = "boardlife" if is_korean else "bgg"
    rating_max = 5 if is_korean else 10
    return {
        "id":          g.pk,
        "title":       g.name,
        "category":    "boardgame",
        "emoji":       "🎲",
        "cat_label":   "보드게임",
        "rating":      g.rating,
        "rating_max":  rating_max,
        "source":      source,
        "players":     g.players_display,
        "time":        g.play_time_display,
        "difficulty":  _normalize_difficulty(g.difficulty),
        "horror":      None,
        "region":      None,
        "tags":        _filter_tags(g.tags)[:4],
        "description": (g.description or "")[:80],
    }


# =====================================================================
# 카테고리 탐색 메인 페이지
# =====================================================================

class ExploreView(View):
    def get(self, request):
        category   = request.GET.get("category", "all")
        search     = request.GET.get("search", "").strip().lower()
        difficulty = request.GET.get("difficulty", "")
        page_num   = int(request.GET.get("page", 1))

        activities = []

        # ── 보드게임 ──────────────────────────────────────
        if category in ("all", "boardgame"):
            qs_all = BoardGame.objects.all()
            if search:
                qs_all = qs_all.filter(name__icontains=search)

            all_games = list(qs_all.order_by("pk"))

            if difficulty:
                all_games = [g for g in all_games
                             if _normalize_difficulty(g.difficulty) == difficulty]

            for g in all_games:
                activities.append(_to_activity_boardgame(g))

        # ── 방탈출 ────────────────────────────────────────
        if category in ("all", "escape"):
            qs = Escape.objects.all()
            if search:
                qs = qs.filter(name__icontains=search)

            for g in qs:

                # 별점 5.0 (만점)은 신뢰도 낮으므로 제외
                if g.rating and round(float(g.rating), 1) == 5.0:
                    continue
                
                norm_diff = _normalize_difficulty(g.difficulty)
                if difficulty and norm_diff != difficulty:
                    continue

                if g.fear_level is None or g.fear_level == 0:
                    horror = "공포 없음"
                elif g.fear_level <= 2:
                    horror = "공포 약함"
                elif g.fear_level <= 3:
                    horror = "공포 있음"
                else:
                    horror = "공포 강함"

                activities.append({
                    "id":          g.pk,
                    "title":       g.name,
                    "category":    "escape",
                    "emoji":       "🔐",
                    "cat_label":   "방탈출",
                    "rating":      g.rating,
                    "rating_max":  5,
                    "source":      "bbabang",
                    "players":     g.players_display,
                    "time":        g.play_time_display,
                    "difficulty":  norm_diff,
                    "horror":      horror,
                    "region":      g.region or None,
                    "tags":        _filter_tags(g.tags)[:4],
                    "description": (g.description or "")[:80],
                })

        # ── 크라임씬 ──────────────────────────────────────
        if category in ("all", "crimescene"):
            qs = CrimeScene.objects.all()
            if search:
                qs = qs.filter(name__icontains=search)

            for g in qs:
                norm_diff = _normalize_difficulty(g.difficulty)
                if difficulty and norm_diff != difficulty:
                    continue

                activities.append({
                    "id":          g.pk,
                    "title":       g.name,
                    "category":    "crimescene",
                    "emoji":       "🕵️",
                    "cat_label":   "크라임씬",
                    "rating":      g.rating,
                    "rating_max":  5,
                    "source":      "crimescene",
                    "players":     g.players_display,
                    "time":        g.play_time_display,
                    "difficulty":  norm_diff,
                    "horror":      None,
                    "region":      None,
                    "tags":        _filter_tags(g.tags)[:4],
                    "description": (g.description or "")[:80],
                })

        # 태그 / 설명 추가 검색
        if search:
            activities = [
                a for a in activities
                if search in a["title"].lower()
                or any(search in t.lower() for t in a["tags"])
                or search in (a["description"] or "").lower()
            ]

        # ── 정렬: 한국어 우선 + 정규화 별점 내림차순 ──────
        activities.sort(key=_sort_key, reverse=True)

        # 페이지네이션
        total     = len(activities)
        paginator = Paginator(activities, PAGE_SIZE)
        page_obj  = paginator.get_page(page_num)

        return render(request, "contents/explore.html", {
            "current_page":   "explore",
            "activities":     list(page_obj.object_list),
            "total":          total,
            "page_obj":       page_obj,
            "page_num":       page_num,
            "has_next":       page_obj.has_next(),
            "has_prev":       page_obj.has_previous(),
            "sel_category":   category,
            "sel_search":     request.GET.get("search", ""),
            "sel_difficulty": difficulty,
        })


# =====================================================================
# 보드게임
# =====================================================================

class BoardGameListView(View):
    def get(self, request):
        qs = BoardGame.objects.all()
        if q := request.GET.get("q", ""):
            qs = qs.filter(name__icontains=q)
        if d := request.GET.get("difficulty", ""):
            qs = qs.filter(difficulty=d)
        if p := request.GET.get("min_players", ""):
            qs = qs.filter(players_max__gte=int(p))
        if t := request.GET.get("max_time", ""):
            qs = qs.filter(play_time__lte=int(t))

        paginator = Paginator(qs, 12)
        page      = paginator.get_page(int(request.GET.get("page", 1)))
        games     = [_serialize_boardgame(g) for g in page.object_list]

        if request.headers.get("Accept") == "application/json":
            return JsonResponse({"results": games, "total_pages": paginator.num_pages, "count": paginator.count})
        return render(request, "contents/boardgame/list.html", {"games": page})


class BoardGameDetailView(View):
    def get(self, request, pk):
        try:
            game = BoardGame.objects.get(pk=pk)
        except BoardGame.DoesNotExist:
            return JsonResponse({"error": "게임을 찾을 수 없습니다."}, status=404)
        data = _serialize_boardgame(game, detail=True)
        if request.headers.get("Accept") == "application/json":
            return JsonResponse(data)
        return render(request, "contents/boardgame/detail.html", {"game": data})


# =====================================================================
# 방탈출
# =====================================================================

class EscapeListView(View):
    def get(self, request):
        qs = Escape.objects.all()
        if q := request.GET.get("q", ""):
            qs = qs.filter(name__icontains=q)
        if r := request.GET.get("region", ""):
            qs = qs.filter(region=r)
        if d := request.GET.get("difficulty", ""):
            qs = qs.filter(difficulty=d)
        if h := request.GET.get("max_horror", ""):
            qs = qs.filter(fear_level__lte=int(h))

        paginator = Paginator(qs, 12)
        page      = paginator.get_page(int(request.GET.get("page", 1)))
        games     = [_serialize_escape(g) for g in page.object_list]

        if request.headers.get("Accept") == "application/json":
            return JsonResponse({"results": games, "total_pages": paginator.num_pages, "count": paginator.count})
        return render(request, "contents/escape/list.html", {"games": page})


class EscapeDetailView(View):
    def get(self, request, pk):
        try:
            game = Escape.objects.get(pk=pk)
        except Escape.DoesNotExist:
            return JsonResponse({"error": "게임을 찾을 수 없습니다."}, status=404)
        data = _serialize_escape(game, detail=True)
        if request.headers.get("Accept") == "application/json":
            return JsonResponse(data)
        return render(request, "contents/escape/detail.html", {"game": data})


# =====================================================================
# 크라임씬
# =====================================================================

class CrimeSceneListView(View):
    def get(self, request):
        qs = CrimeScene.objects.all()
        if q := request.GET.get("q", ""):
            qs = qs.filter(name__icontains=q)
        if d := request.GET.get("difficulty", ""):
            qs = qs.filter(difficulty=d)
        if p := request.GET.get("min_players", ""):
            qs = qs.filter(players_max__gte=int(p))
        if t := request.GET.get("max_time", ""):
            qs = qs.filter(play_time__lte=int(t))

        paginator = Paginator(qs, 12)
        page      = paginator.get_page(int(request.GET.get("page", 1)))
        games     = [_serialize_crimescene(g) for g in page.object_list]

        if request.headers.get("Accept") == "application/json":
            return JsonResponse({"results": games, "total_pages": paginator.num_pages, "count": paginator.count})
        return render(request, "contents/crimescene/list.html", {"games": page})


class CrimeSceneDetailView(View):
    def get(self, request, pk):
        try:
            game = CrimeScene.objects.get(pk=pk)
        except CrimeScene.DoesNotExist:
            return JsonResponse({"error": "게임을 찾을 수 없습니다."}, status=404)
        data = _serialize_crimescene(game, detail=True)
        if request.headers.get("Accept") == "application/json":
            return JsonResponse(data)
        return render(request, "contents/crimescene/detail.html", {"game": data})


# =====================================================================
# 직렬화 헬퍼
# =====================================================================

def _serialize_boardgame(g, detail=False):
    is_korean = _is_korean(g.name)
    data = {
        "id": g.pk, "category": "boardgame",
        "name": g.name, "rating": g.rating,
        "rating_max": 5 if is_korean else 10,
        "source": "boardlife" if is_korean else "bgg",
        "players": g.players_display, "play_time": g.play_time_display,
        "difficulty": _normalize_difficulty(g.difficulty),
        "tags": g.tags if isinstance(g.tags, list) else [],
        "publisher": g.publisher, "designer": g.designer,
        # mechanism 제거
        "description": g.description or "",   # ← 설명 항상 포함
    }
    if detail:
        data["reviews"] = []   # ← 리뷰 제거
    return data


def _serialize_escape(g, detail=False):
    if g.fear_level is None or g.fear_level == 0:
        horror_text = "공포 없음"
    elif g.fear_level <= 2:
        horror_text = "약함"
    elif g.fear_level <= 3:
        horror_text = "중간"
    else:
        horror_text = "강함"

    data = {
        "id": g.pk, "category": "escape",
        "name": g.name, "rating": g.rating,
        "rating_max": 5, "source": "bbabang",
        "players": g.players_display, "play_time": g.play_time_display,
        "difficulty": _normalize_difficulty(g.difficulty),
        "tags": g.tags if isinstance(g.tags, list) else [],
        "region": g.region, "brand": g.brand,
        "theme": g.theme, "horror_level": horror_text, "fear_level": g.fear_level,
        "description": g.description or "",   # ← 설명 항상 포함
    }
    if detail:
        data["reviews"] = []   # ← 리뷰 제거
    return data


def _serialize_crimescene(g, detail=False):
    data = {
        "id": g.pk, "category": "crimescene",
        "name": g.name, "rating": g.rating,
        "rating_max": 5, "source": "crimescene",
        "players": g.players_display, "play_time": g.play_time_display,
        "difficulty": _normalize_difficulty(g.difficulty),
        "tags": g.tags if isinstance(g.tags, list) else [],
        "series": g.series, "maker": g.maker,
        "publisher": g.publisher, "publisher_kr": g.publisher_kr,
        "description": g.description or "",   # ← 설명 항상 포함
    }
    if detail:
        data["reviews"] = []   # ← 리뷰 제거
    return data