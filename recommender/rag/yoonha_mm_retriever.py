"""
yoonha_mm_retriever.py
머더미스터리 통합 검색 (머미나우 + 머더미스터리로그)

====================================================================
[역할]
    머미나우(murmynow)와 머더미스터리로그(murdermysterylog) 두 소스를
    대상으로 소스별 독립적인 BM25 + FAISS RRF 검색을 수행한 뒤,
    크로스-소스 RRF로 최종 순위를 산출한다.

[방식: 소스별 RRF 후 크로스-소스 재융합]
    단순 병합(Simple Merge) 대신 이 방식을 선택한 이유:
      - 머미나우(4,534개) vs 머더로그(281개) 규모 불균형
      - 단순 병합 시 머미나우가 BM25 corpus를 압도 -> 머더로그 결과가 묻힘
      - 소스별 독립 RRF로 각 corpus 내 최선 결과를 뽑은 뒤,
        크로스-소스 RRF로 재융합하면 두 소스가 균형 있게 반영됨

[구조]
    1. 소스별 데이터 로드 (FAISS index + meta JSON + 독립 BM25 corpus)
    2. hard_filter(): 인원/시간/scene_category 하드 필터
    3. 소스별 BM25 + FAISS -> _rrf_fuse_source() -> 소스 내 top-N
    4. _cross_source_rrf(): 두 소스 결과를 크로스-소스 RRF로 최종 융합

[머미나우 meta JSON 구조]
    faiss_murmynow_meta.json에는 stats(239개)와 review(4295개)가 혼재.
      - stats: min/max_players, difficulty, rating 등 메타 보유
      - review: review_rating, review_difficulty만 보유 (필터 필드 없음)
    BM25 corpus와 검색 결과에는 stats 항목만 사용한다.
      이유 1: review 항목은 min/max_players 등 필터 필드가 없어
              hard_filter를 우회할 수 있음
      이유 2: BM25 corpus를 review 4295개로 늘리면 IDF 계산이 왜곡됨
    FAISS 인덱스는 4534개 벡터를 보유하므로 dense 검색은 _mn_items 전체로
    인덱스 매핑하되, hard_filter에서 review 항목(type="review")을 차단한다.

[소스별 필드 차이]
    머미나우  : name, min_time/max_time(범위형), scene_category,
               difficulty(이산형 1~4), rating(0~5), image_url
    머더로그  : title(또는 name), play_time(단일값), rating(0~5),
               reviews(텍스트, || 구분)

[하드 필터 필드 매핑 — query_transformer 기준]
    players  <- group.headcount
    max_time <- group.play_time  (murdermystery 전용 키)

[가중치 설계]
    rating        : 두 소스 모두 0~5, 정규화 후 동일 기준 적용
    difficulty    : 머미나우 전용 이산형(1~4). 빠방 difficulty(연속형 0~5)와 혼용 금지
    horror        : 머더로그 전용. 0~5, 높을수록 공포 강함
    scene_category: 머미나우 전용 소프트 부스트

[개선 포인트]
    1. 크로스-소스 RRF k=60 고정 -> 소스 간 비율 조정 여지 있음
    2. 머더로그 horror 필드 범위(0~5) 확인 필요
    3. emotion_tags는 tag_filter 단계에서 별도 처리됨
====================================================================
"""

import json
import re
import math
import faiss
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

# =========================================================
# 데이터 로드
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "04_vectorstore"

# 머더미스터리로그
_mm_index = faiss.read_index(str(DATA_DIR / "faiss_murdermysterylog.index"))
with open(DATA_DIR / "faiss_murdermysterylog_meta.json", "r", encoding="utf-8") as f:
    _mm_items = json.load(f)
for item in _mm_items:
    item["source"] = "murdermysterylog"
    if "name" in item and "title" not in item:
        item["title"] = item["name"]

# 머미나우 (stats + review 혼재)
_mn_index = faiss.read_index(str(DATA_DIR / "faiss_murmynow.index"))
with open(DATA_DIR / "faiss_murmynow_meta.json", "r", encoding="utf-8") as f:
    _mn_items = json.load(f)
for item in _mn_items:
    item["source"] = "murmynow"
    if "name" in item and "title" not in item:
        item["title"] = item["name"]

# stats 전용 항목 (239개) — BM25 corpus 및 검색 결과에 사용
# FAISS 인덱스는 4534개 기준이므로 dense 검색엔 _mn_items(전체) 사용
_mn_stats_items = [item for item in _mn_items if item.get("type", "stats") != "review"]

print(f"[mm_retriever] 머더미스터리로그: {len(_mm_items)}개 (dim={_mm_index.d})")
print(
    f"[mm_retriever] 머미나우 전체: {len(_mn_items)}개 / "
    f"stats 전용: {len(_mn_stats_items)}개 (dim={_mn_index.d})"
)


# =========================================================
# BM25 준비 — 소스별 독립 corpus
# =========================================================
def _make_searchable_text(item: dict) -> str:
    """BM25 검색용 텍스트 조합. 소스 공통 처리."""
    parts = [str(item.get("title") or item.get("name", ""))]

    if item.get("description"):
        parts.append(str(item["description"])[:500])

    # 머미나우 전용 필드
    if item.get("author"):
        parts.append(str(item["author"]))
    if item.get("publisher"):
        parts.append(str(item["publisher"]))
    if item.get("scene_category"):
        parts.append(str(item["scene_category"]))

    # 머더로그 전용 필드
    if item.get("시리즈"):
        parts.append(str(item["시리즈"]))
    if item.get("제작"):
        parts.append(str(item["제작"]))
    if item.get("reviews"):
        # || 구분자로 연결된 리뷰 텍스트를 BM25 corpus에 포함
        parts.append(str(item["reviews"]))

    for tag in item.get("emotion_tags", []):
        parts.append(str(tag))

    return " ".join(parts)


# 머더로그: 281개
_mm_corpus = [_make_searchable_text(s) for s in _mm_items]
_mm_tokenized = [c.lower().split() for c in _mm_corpus]
_mm_bm25 = BM25Okapi(_mm_tokenized)

# 머미나우: stats 전용 239개 (review 제외 -> IDF 왜곡 방지)
_mn_corpus = [_make_searchable_text(s) for s in _mn_stats_items]
_mn_tokenized = [c.lower().split() for c in _mn_corpus]
_mn_bm25 = BM25Okapi(_mn_tokenized)

print(
    f"[mm_retriever] BM25 준비 완료 — "
    f"머더로그: {len(_mm_corpus)}개, 머미나우(stats): {len(_mn_corpus)}개"
)


# =========================================================
# 메타데이터 정규화 유틸
# =========================================================
def _as_number(value, default=None):
    """문자열/숫자 메타데이터를 float로 안전 변환. None은 0으로 보지 않는다."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned or cleaned.lower() in {"none", "null", "nan", "?", "-"}:
            return default
        m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return default
    return default


def _as_int(value, default=None):
    num = _as_number(value, default=None)
    return int(num) if num is not None else default


def _contains(value, target: str) -> bool:
    if not target or value is None:
        return False
    return str(target).lower() in str(value).lower()


# =========================================================
# 하드 필터
# =========================================================
def hard_filter(item: dict, query_filter: dict) -> bool:
    """
    True = 통과.

    시간 필드 소스별 차이:
      머미나우  -> max_time (범위형 max_time/min_time 중 max 사용)
      머더로그  -> play_time (단일값), 없으면 max_time 대체 시도

    query_filter 키:
      players  : 인원 하드 필터
      max_time : 최대 플레이 시간(분) 하드 필터
      scene_category : 작품 유형 하드 필터 (머미나우 전용 실데이터)
    """
    # -- review 항목 제거 (안전망: 필터/점수 필드가 없어 결과 품질 저하) --
    if item.get("type") == "review":
        return False

    # -- 인원 필터 --
    players = _as_int(query_filter.get("players"), default=None)
    if players is not None:
        max_p = _as_int(item.get("max_players"), default=999)
        min_p = _as_int(item.get("min_players"), default=0)
        if players > max_p or players < min_p:
            return False

    # -- 시간 필터 --
    max_time = _as_int(query_filter.get("max_time"), default=None)
    if max_time is not None:
        source = item.get("source", "")
        if source == "murmynow":
            item_time = _as_int(item.get("max_time"), default=None)
        else:  # murdermysterylog
            item_time = _as_int(item.get("play_time"), default=None)
            if item_time is None:
                item_time = _as_int(item.get("max_time"), default=None)
        if item_time is not None and item_time > 0 and item_time > max_time:
            return False

    # -- scene_category 필터 (머미나우에만 실데이터 존재) --
    scene_category = query_filter.get("scene_category")
    if scene_category:
        item_scene = (
            item.get("scene_category")
            or item.get("유형")
            or ""
        )
        if item_scene and not _contains(item_scene, scene_category):
            return False

    return True


# =========================================================
# 메타데이터 가중치
# =========================================================
def _metadata_weight(item: dict, query_filter: dict) -> float:
    """
    소스별 메타데이터 기반 2차 가중치.

    공통:
      rating(0~5)  : 높을수록 가산. None은 데이터 없음으로 처리
      인원 정확도  : min/max 범위가 좁을수록 보너스

    머미나우 전용:
      difficulty(이산형 1~4): 1=쉬워요 / 2=보통 / 3=어려워요 / 4=매우 어려워요
        - 빠방 difficulty(연속형 0~5)와 스케일 다름 -> 혼용 금지
      scene_category: 조건 매칭 시 소프트 부스트

    머더로그 전용:
      horror(0~5)  : horror_pref에 따라 가산/감산
      reviews 텍스트: 감성 키워드 빈도 + 리뷰 양 신뢰도 반영
    """
    score = 0.0
    source = item.get("source", "murdermysterylog")

    # -- 1. rating (0~5, 두 소스 동일 스케일) --
    rating = _as_number(item.get("rating"), default=None)
    if rating is not None and rating > 0:
        score += min(rating / 5.0, 1.0) * 12.0

    # -- 2. difficulty preference (머미나우 전용 이산형 1~4) --
    if source == "murmynow":
        difficulty = _as_number(item.get("difficulty"), default=None)
        pref = query_filter.get("difficulty_pref") or query_filter.get("weight_pref")
        if pref and difficulty is not None:
            if pref == "light":
                score += max(0.0, (3.0 - difficulty) / 2.0) * 6.0
                if difficulty >= 3:
                    score -= min((difficulty - 2.0) * 4.0, 8.0)
            elif pref == "medium":
                score += max(0.0, 1.0 - abs(difficulty - 2.0) / 2.0) * 5.0
            elif pref == "heavy":
                score += max(0.0, (difficulty - 2.0) / 2.0) * 6.0
                if difficulty <= 2:
                    score -= min((3.0 - difficulty) * 4.0, 8.0)

    # -- 3. horror preference (머더로그 전용, 0~5) --
    if source == "murdermysterylog":
        horror = _as_number(item.get("horror"), default=None)
        horror_pref = query_filter.get("horror_pref")
        if horror_pref and horror is not None:
            if horror_pref == "low":
                score += max(0.0, (5.0 - horror) / 5.0) * 5.0
            elif horror_pref == "medium":
                score += max(0.0, 1.0 - abs(horror - 2.5) / 2.5) * 4.0
            elif horror_pref == "high":
                score += max(0.0, horror / 5.0) * 5.0

    # -- 4. scene_category 소프트 부스트 (머미나우 전용 실데이터) --
    if query_filter.get("scene_category"):
        item_scene = (
            item.get("scene_category")
            or item.get("유형")
            or ""
        )
        if _contains(item_scene, query_filter["scene_category"]):
            score += 5.0

    # -- 5. 추천 인원 정확도 보너스 --
    players = _as_int(query_filter.get("players"), default=None)
    if players is not None:
        min_p = _as_int(item.get("min_players"), default=None)
        max_p = _as_int(item.get("max_players"), default=None)
        if min_p is not None and max_p is not None and min_p > 0:
            player_range = max_p - min_p
            if player_range <= 2:
                score += 4.0
            elif player_range < 6:
                score += 2.0

    # -- 6. reviews 텍스트 감성/양 보정 (머더로그 전용) --
    if source == "murdermysterylog":
        reviews_text = str(item.get("reviews", "") or "")
        if reviews_text:
            positive_kw = ["재밌", "재미있", "명작", "수작", "추천", "최고", "몰입", "만족"]
            negative_kw = ["별로", "실망", "지루", "노잼", "비추", "최악", "아쉬"]
            pos_hits = sum(reviews_text.lower().count(kw) for kw in positive_kw)
            neg_hits = sum(reviews_text.lower().count(kw) for kw in negative_kw)
            sentiment_score = min(pos_hits * 0.03 - neg_hits * 0.05, 3.0)
            score += max(sentiment_score, -3.0)
            review_count = reviews_text.count("||") + 1
            score += min(math.log1p(review_count) / math.log1p(50), 1.0) * 2.0

    return score


# =========================================================
# 단일 소스 BM25 / Dense / RRF
# =========================================================
def _bm25_search_source(
    items: list,
    bm25: BM25Okapi,
    query_text: str,
    query_filter: dict,
    topk: int = 200,
) -> dict:
    """단일 소스 BM25 검색. 반환: {title: {item, rank, bm25_score}}"""
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1]
    results: dict = {}
    rank = 0
    for idx in top_idx:
        if idx >= len(items):
            continue
        item = items[idx]
        if not hard_filter(item, query_filter):
            continue
        title = item.get("title") or item.get("name", "")
        if title not in results:
            rank += 1
            results[title] = {
                "item": item,
                "rank": rank,
                "bm25_score": float(scores[idx]),
            }
            if rank >= topk:
                break
    return results


def _dense_search_source(
    items: list,
    index: faiss.Index,
    query_vector: np.ndarray,
    query_filter: dict,
    topk: int = 200,
) -> dict:
    """단일 소스 FAISS Dense 검색. 반환: {title: {item, rank, l2_dist}}"""
    if query_vector.shape[1] != index.d:
        raise ValueError(
            f"벡터 dim 불일치: {query_vector.shape[1]} != {index.d}"
        )
    D, I = index.search(query_vector, topk * 3)
    results: dict = {}
    rank = 0
    for i, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(items):
            continue
        item = items[idx]
        if not hard_filter(item, query_filter):
            continue
        title = item.get("title") or item.get("name", "")
        if title not in results:
            rank += 1
            results[title] = {
                "item": item,
                "rank": rank,
                "l2_dist": float(D[0][i]),
            }
            if rank >= topk:
                break
    return results


def _rrf_fuse_source(
    bm25_results: dict,
    dense_results: dict,
    query_filter: dict,
    topk: int,
    k: int = 60,
    scale: float = 1000.0,
) -> list:
    """
    단일 소스 내 BM25 + Dense RRF 융합 + 메타데이터 가중치.

    total_score = rrf_score x scale + meta_score
    scale=1000: rrf_score(0.001~0.03) 범위를 meta_score(0~30)와 맞춤.
    BM25에만 등장한 아이템은 dense 미등장 패널티(x 0.7) 적용.
    """
    all_titles = set(list(bm25_results.keys()) + list(dense_results.keys()))
    scored = []

    for title in all_titles:
        bm25_data = bm25_results.get(title)
        dense_data = dense_results.get(title)
        bm25_rank = bm25_data["rank"] if bm25_data else 999
        dense_rank = dense_data["rank"] if dense_data else 999

        if bm25_rank == 999 and dense_rank == 999:
            continue

        rrf_score = 1 / (k + bm25_rank) + 1 / (k + dense_rank)
        if dense_rank == 999:
            rrf_score *= 0.7  # dense 미등장 패널티

        item = (bm25_data or dense_data)["item"]
        meta_score = _metadata_weight(item, query_filter)
        total_score = rrf_score * scale + meta_score

        item_copy = item.copy()
        item_copy["rrf_score"] = round(rrf_score, 6)
        item_copy["meta_score"] = round(meta_score, 2)
        item_copy["total_score"] = round(total_score, 2)
        item_copy["bm25_rank"] = bm25_rank
        item_copy["dense_rank"] = dense_rank
        scored.append(item_copy)

    scored.sort(key=lambda x: x["total_score"], reverse=True)
    return scored[:topk]


# =========================================================
# 크로스-소스 RRF 재융합
# =========================================================
def _cross_source_rrf(
    mm_results: list,
    mn_results: list,
    topk: int,
    k: int = 60,
) -> list:
    """
    소스별 RRF 결과를 크로스-소스 RRF로 재융합.

    각 소스 결과 리스트에서의 순위(position + 1)를 rank로 사용.
    두 소스의 corpus 크기 불균형을 해소하기 위해,
    각 소스 내부 순위만으로 최종 점수를 결정한다.

    cross_rrf_score = 1/(k + mm_rank) + 1/(k + mn_rank)

    같은 제목이 두 소스에 모두 존재하면 두 순위를 모두 반영.
    한 소스에만 있으면 나머지 rank=999로 처리.
    """
    mm_rank_map: dict = {
        (item.get("title") or item.get("name", "")): (rank + 1, item)
        for rank, item in enumerate(mm_results)
    }
    mn_rank_map: dict = {
        (item.get("title") or item.get("name", "")): (rank + 1, item)
        for rank, item in enumerate(mn_results)
    }

    all_titles = set(list(mm_rank_map.keys()) + list(mn_rank_map.keys()))
    scored = []

    for title in all_titles:
        mm_entry = mm_rank_map.get(title)
        mn_entry = mn_rank_map.get(title)
        mm_rank = mm_entry[0] if mm_entry else 999
        mn_rank = mn_entry[0] if mn_entry else 999

        # max 방식: 겹치는 항목이 두 소스 합산으로 이중 부스트 받지 않도록 함.
        # sum 방식(1/(k+mm) + 1/(k+mn))은 겹치는 166개가 상위 독점 → 머미나우 전용 73개 노출 차단.
        # max 방식은 각 아이템이 "가장 잘 맞는 소스에서의 순위"로만 경쟁하므로
        # 소스 간 공정한 경쟁이 보장됨.
        cross_rrf = max(1 / (k + mm_rank), 1 / (k + mn_rank))

        # 두 소스 모두에 있으면 머더로그 item 우선 (메타 더 풍부)
        item = (mm_entry or mn_entry)[1]
        item_copy = item.copy()
        item_copy["cross_rrf_score"] = round(cross_rrf, 6)
        item_copy["mm_rank"] = mm_rank
        item_copy["mn_rank"] = mn_rank
        scored.append(item_copy)

    scored.sort(key=lambda x: x["cross_rrf_score"], reverse=True)
    return scored[:topk]


# =========================================================
# 앵커 임베딩 유틸
# =========================================================
def get_embedding(titles: list) -> np.ndarray:
    """
    타이틀 리스트 -> 평균 임베딩 벡터 반환 (shape: (1, dim)).
    머더로그 -> 머미나우(stats) 순으로 탐색.
    미발견 시 머더로그 인덱스 0번 벡터 사용.
    """
    embeddings = []
    for title in titles:
        found = False
        for index, meta in [(_mm_index, _mm_items), (_mn_index, _mn_stats_items)]:
            for i, s in enumerate(meta):
                if s.get("title") == title or s.get("name") == title:
                    embeddings.append(index.reconstruct(i))
                    found = True
                    break
            if found:
                break
    if embeddings:
        return np.mean(embeddings, axis=0).reshape(1, -1).astype(np.float32)
    return _mm_index.reconstruct(0).reshape(1, -1).astype(np.float32)


# =========================================================
# 공개 인터페이스
# =========================================================
def retrieve(
    query_text: str,
    query_filter: dict,
    query_vector: np.ndarray,
    topk: int = 50,
) -> list:
    """
    소스별 RRF 후 크로스-소스 재융합 (주력 검색).

    흐름:
      1. 머더로그: BM25 + FAISS -> RRF -> intermediate_topk개
      2. 머미나우: BM25(stats) + FAISS(전체, review 차단) -> RRF -> intermediate_topk개
      3. 크로스-소스 RRF -> 최종 topk개
    """
    intermediate_topk = max(topk * 2, 100)

    # -- 머더로그 소스 --
    mm_bm25 = _bm25_search_source(
        _mm_items, _mm_bm25, query_text, query_filter, topk=200
    )
    mm_dense = _dense_search_source(
        _mm_items, _mm_index, query_vector, query_filter, topk=200
    )
    mm_results = _rrf_fuse_source(
        mm_bm25, mm_dense, query_filter, topk=intermediate_topk, scale=1000.0
    )

    # -- 머미나우 소스 --
    # BM25: corpus가 stats 239개 기준이므로 _mn_stats_items의 인덱스 매핑이 맞음
    # Dense: FAISS 인덱스가 4534개(stats+review) 기준이므로 _mn_items로 매핑,
    #        hard_filter에서 review 항목(type="review")을 차단
    mn_bm25 = _bm25_search_source(
        _mn_stats_items, _mn_bm25, query_text, query_filter, topk=200
    )
    mn_dense = _dense_search_source(
        _mn_items, _mn_index, query_vector, query_filter, topk=200
    )
    mn_results = _rrf_fuse_source(
        mn_bm25, mn_dense, query_filter, topk=intermediate_topk, scale=1000.0
    )

    # -- 크로스-소스 RRF 재융합 --
    return _cross_source_rrf(mm_results, mn_results, topk=topk)


def retrieve_bm25(
    query_text: str,
    query_filter: dict,
    topk: int = 50,
) -> list:
    """BM25 단독 검색 (두 소스 결과 라운드로빈 인터리브)."""
    mm_res = _bm25_search_source(
        _mm_items, _mm_bm25, query_text, query_filter, topk=topk
    )
    mn_res = _bm25_search_source(
        _mn_stats_items, _mn_bm25, query_text, query_filter, topk=topk
    )
    mm_ranked = [d["item"] for d in sorted(mm_res.values(), key=lambda x: x["rank"])]
    mn_ranked = [d["item"] for d in sorted(mn_res.values(), key=lambda x: x["rank"])]

    merged = []
    for i in range(max(len(mm_ranked), len(mn_ranked))):
        if i < len(mm_ranked):
            merged.append(mm_ranked[i])
        if i < len(mn_ranked):
            merged.append(mn_ranked[i])
    return merged[:topk]


def retrieve_dense(
    query_vector: np.ndarray,
    query_filter: dict,
    topk: int = 50,
) -> list:
    """Dense 단독 검색 (두 소스 결과 라운드로빈 인터리브)."""
    mm_res = _dense_search_source(
        _mm_items, _mm_index, query_vector, query_filter, topk=topk
    )
    mn_res = _dense_search_source(
        _mn_items, _mn_index, query_vector, query_filter, topk=topk
    )
    mm_ranked = [d["item"] for d in sorted(mm_res.values(), key=lambda x: x["rank"])]
    mn_ranked = [d["item"] for d in sorted(mn_res.values(), key=lambda x: x["rank"])]

    merged = []
    for i in range(max(len(mm_ranked), len(mn_ranked))):
        if i < len(mm_ranked):
            merged.append(mm_ranked[i])
        if i < len(mn_ranked):
            merged.append(mn_ranked[i])
    return merged[:topk]


def retrieve_vanilla(
    query_filter: dict,
    topk: int = 50,
) -> list:
    """Vanilla 검색 — 두 소스 합산 후 평점 내림차순."""
    all_items = [
        item.copy()
        for source_items in [_mm_items, _mn_stats_items]
        for item in source_items
        if hard_filter(item, query_filter)
    ]
    all_items.sort(key=lambda x: _as_number(x.get("rating"), default=0.0), reverse=True)
    return all_items[:topk]
