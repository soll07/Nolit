"""
recommender/rag/yoonha_query_transformer.py

[ 역할 ]
  그룹 조건(group dict) + 자연어 입력(user_text) → 4종 검색 쿼리로 변환

[ 출력 4종 ]
  query_text    : BM25 키워드 검색용 자연어 쿼리 문자열
  query_filter  : hard_filter 조건 dict (retriever의 하드 필터링에 사용)
  emotion_tags  : tag_filter용 감정 태그 리스트
  anchor_titles : dense embedding용 앵커 타이틀 리스트
"""

from __future__ import annotations


WEIGHT_MAX_MAP = {
    "light": 2.5,
    "medium": 3.5,
}

WEIGHT_TEXT_MAP = {
    "light":  "가벼운 입문 간단 쉬운",
    "medium": "중간 보통 중급",
    "heavy":  "무거운 복잡 전략 고급 어려운",
}

CATEGORY_TEXT_MAP = {
    "Strategy":    "전략 strategy",
    "Economic":    "경제 economic",
    "Party":       "파티 party",
    "War":         "전쟁 war",
    "Family":      "가족 family",
    "Abstract":    "추상 abstract",
    "Cooperative": "협력 cooperative",
    "Deduction":   "추리 deduction",
    "Card Game":   "카드게임 card",
    "Thematic":    "테마 thematic",
}

MECHANISM_TEXT_MAP = {
    "Worker Placement": "일꾼배치 worker placement",
    "Deck Building":    "덱빌딩 deck building",
    "Engine Building":  "엔진빌딩 engine building",
    "Area Control":     "지역장악 area control",
    "Market":           "시장 market",
    "Drafting":         "드래프팅 drafting",
    "Cooperative Game": "협력 cooperative",
    "Tile Placement":   "타일배치 tile placement",
}

HORROR_EMOTION_MAP = {
    0: ["공포없음"],
    1: [],
    2: [],
}

ESCAPE_PREF_TEXT_MAP = {
    "prefer_puzzle":     "퍼즐 문제 힌트",
    "prefer_story":      "스토리 서사 몰입",
    "prefer_interior":   "인테리어 예쁜 분위기",
    "prefer_production": "연출 장치 퀄리티",
}

WEIGHT_EMOTION_MAP = {
    "light":  ["입문용", "가볍게즐길수있음"],
    "medium": [],
    "heavy":  [],
}

BOARDGAME_ANCHORS = {
    "Strategy":    ["Brass: Birmingham", "Twilight Struggle", "Terra Mystica"],
    "Economic":    ["Brass: Birmingham", "Ark Nova", "Terraforming Mars"],
    "Party":       ["Codenames", "Dixit", "Wavelength"],
    "Cooperative": ["Pandemic", "Spirit Island", "Arkham Horror"],
    "Deduction":   ["Mysterium", "Codenames", "Sherlock Holmes"],
    "Family":      ["Ticket to Ride", "Carcassonne", "Wingspan"],
    "War":         ["Twilight Struggle", "War of the Ring", "Memoir 44"],
}

MM_ANCHORS = {
    "default": ["구두룡 저택의 살인"],
}


def _normalize_difficulty_pref(group):
    return group.get("difficulty_pref") or group.get("weight_pref")


def _build_query_text(user_text, group, category):
    parts = [user_text] if user_text else []

    headcount = group.get("headcount")
    if headcount:
        parts.append(str(headcount) + "인")

    weight_pref = group.get("weight_pref")
    if weight_pref and weight_pref in WEIGHT_TEXT_MAP:
        parts.append(WEIGHT_TEXT_MAP[weight_pref])

    if category == "boardgame":
        cat = group.get("category")
        if cat and cat in CATEGORY_TEXT_MAP:
            parts.append(CATEGORY_TEXT_MAP[cat])
        mech = group.get("mechanism")
        if mech and mech in MECHANISM_TEXT_MAP:
            parts.append(MECHANISM_TEXT_MAP[mech])

    if category == "murdermystery":
        parts.append("추리 크라임씬 머더미스터리")
        horror_tolerance = group.get("horror_tolerance", 2)
        if horror_tolerance == 0:
            parts.append("공포없음 안무서운")
        elif horror_tolerance == 1:
            parts.append("약간 공포")
        elif horror_tolerance == 2:
            parts.append("공포 가능 호러")
        difficulty_pref = _normalize_difficulty_pref(group)
        if difficulty_pref and difficulty_pref in WEIGHT_TEXT_MAP:
            parts.append(WEIGHT_TEXT_MAP[difficulty_pref])
        if group.get("scene_category"):
            parts.append(str(group["scene_category"]))

    if category == "escape":
        parts.append("방탈출 테마")
        if group.get("location"):
            parts.append(str(group["location"]))
        elif group.get("area"):
            parts.append(str(group["area"]))
        horror_tolerance = group.get("horror_tolerance")
        if horror_tolerance == 0:
            parts.append("안 무서운 공포 없는 순한")
        elif horror_tolerance == 1:
            parts.append("약간 무서운")
        elif horror_tolerance == 2:
            parts.append("무서운 공포 호러")
        difficulty_pref = group.get("difficulty_pref") or group.get("weight_pref")
        if difficulty_pref in ("light", "low"):
            parts.append("쉬운 입문 초보")
        elif difficulty_pref in ("heavy", "high"):
            parts.append("어려운 고난도 빡센")
        for pref_key, text in ESCAPE_PREF_TEXT_MAP.items():
            if group.get(pref_key):
                parts.append(text)

    return " ".join(parts)


def _build_query_filter(group, category):
    query_filter = {}

    headcount = group.get("headcount")
    if headcount:
        query_filter["players"] = headcount

    play_time = group.get("play_time")
    if play_time:
        if category == "boardgame":
            query_filter["playing_time"] = play_time
        elif category == "escape":
            query_filter["playing_time"] = play_time
        else:
            query_filter["max_time"] = play_time

    weight_pref = group.get("weight_pref")
    if weight_pref:
        query_filter["weight_pref"] = weight_pref
        if group.get("strict_weight_filter"):
            query_filter["strict_weight_filter"] = True
            if weight_pref in WEIGHT_MAX_MAP:
                query_filter["weight_max"] = WEIGHT_MAX_MAP[weight_pref]

    if category == "boardgame":
        if group.get("category"):
            query_filter["category"] = group["category"]
        if group.get("mechanism"):
            query_filter["mechanism"] = group["mechanism"]
        query_filter["source_pref"] = (
            group.get("source_pref")
            or group.get("source_preference")
            or "korean"
        )

    if category == "murdermystery":
        difficulty_pref = _normalize_difficulty_pref(group)
        if difficulty_pref:
            query_filter["difficulty_pref"] = difficulty_pref
        if group.get("horror_tolerance") is not None:
            query_filter["horror_tolerance"] = group["horror_tolerance"]
            query_filter["horror_pref"] = (
                "low" if group["horror_tolerance"] == 0
                else "medium" if group["horror_tolerance"] == 1
                else "high"
            )
        if group.get("scene_category"):
            query_filter["scene_category"] = group["scene_category"]
        if group.get("area"):
            query_filter["area"] = group["area"]
        if group.get("location"):
            query_filter["location"] = group["location"]

    if category == "escape":
        if group.get("area"):
            query_filter["area"] = group["area"]
        if group.get("location"):
            query_filter["location"] = group["location"]
        if group.get("price"):
            query_filter["price"] = group["price"]
        if group.get("horror_tolerance") is not None:
            query_filter["horror_pref"] = (
                "low" if group["horror_tolerance"] == 0
                else "medium" if group["horror_tolerance"] == 1
                else "high"
            )
        difficulty_pref = group.get("difficulty_pref") or group.get("weight_pref")
        if difficulty_pref:
            query_filter["difficulty_pref"] = difficulty_pref
        for pref_key in ("prefer_puzzle", "prefer_story", "prefer_interior", "prefer_production"):
            if group.get(pref_key):
                query_filter[pref_key] = True

    return query_filter


def _build_emotion_tags(group):
    tags = []

    horror_tolerance = group.get("horror_tolerance", 2)
    tags.extend(HORROR_EMOTION_MAP.get(horror_tolerance, []))

    weight_pref = group.get("weight_pref")
    if weight_pref:
        tags.extend(WEIGHT_EMOTION_MAP.get(weight_pref, []))

    relation = group.get("relation")
    if relation == "first_meeting":
        tags.extend(["처음만나는사이추천", "분위기좋음", "대화유도"])
    elif relation == "couple":
        tags.extend(["데이트추천", "분위기좋음"])
    elif relation == "friend":
        tags.extend(["웃음", "친목용"])
    elif relation == "coworker":
        tags.extend(["입문용", "가볍게즐길수있음"])

    if group.get("prefer_puzzle"):
        tags.append("퍼즐집중")
    if group.get("prefer_story"):
        tags.append("스토리몰입")
    if group.get("prefer_interior"):
        tags.append("분위기좋음")
    if group.get("prefer_production"):
        tags.append("연출퀄리티")

    return list(dict.fromkeys(tags))


def _build_anchor_titles(group, category):
    if category == "boardgame":
        cat = group.get("category")
        if cat and cat in BOARDGAME_ANCHORS:
            return BOARDGAME_ANCHORS[cat]
        return []

    if category == "escape":
        return []

    return MM_ANCHORS.get("default", [])


def transform(user_text, group, category):
    if category not in ("boardgame", "murdermystery", "escape"):
        raise ValueError(
            f"알 수 없는 category: {category!r}. 'boardgame', 'murdermystery', 'escape' 중 사용."
        )

    return {
        "query_text":    _build_query_text(user_text, group, category),
        "query_filter":  _build_query_filter(group, category),
        "emotion_tags":  _build_emotion_tags(group),
        "anchor_titles": _build_anchor_titles(group, category),
    }
