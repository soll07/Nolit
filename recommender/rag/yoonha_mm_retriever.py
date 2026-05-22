"""
recommender/rag/yoonha_mm_retriever.py
머더미스터리 통합 검색 (머미나우 + 머더미스터리로그)

====================================================================
[역할]
    머미나우(murmynow)와 머더미스터리로그(murdermysterylog) 두 소스를
    대상으로 소스별 독립적인 BM25 + Pinecone RRF 검색을 수행한 뒤,
    크로스-소스 RRF로 최종 순위를 산출한다.

[방식: 소스별 RRF 후 크로스-소스 재융합]
    단순 병합(Simple Merge) 대신 이 방식을 선택한 이유:
      - 머미나우(4,534개) vs 머더로그(281개) 규모 불균형
      - 단순 병합 시 머미나우가 BM25 corpus를 압도 -> 머더로그 결과가 묻힘
      - 소스별 독립 RRF로 각 corpus 내 최선 결과를 뽑은 뒤,
        크로스-소스 RRF로 재융합하면 두 소스가 균형 있게 반영됨

[구조]
    1. 소스별 데이터 로드 (Pinecone index + 독립 BM25 corpus)
    2. hard_filter(): 인원/시간/scene_category 하드 필터
    3. 소스별 BM25 + Pinecone -> _rrf_fuse_source() -> 소스 내 top-N
    4. _cross_source_rrf(): 두 소스 결과를 크로스-소스 RRF로 최종 융합

[머미나우 메타데이터 구조]
    Pinecone murmynow 네임스페이스에는 stats(239개)와 review(4295개)가 혼재.
      - stats: min/max_players, difficulty, rating 등 메타 보유
      - review: review_rating, review_difficulty만 보유 (필터 필드 없음)
    BM25 corpus와 검색 결과에는 stats 항목만 사용한다.
      이유 1: review 항목은 min/max_players 등 필터 필드가 없어
              hard_filter를 우회할 수 있음
      이유 2: BM25 corpus를 review 4295개로 늘리면 IDF 계산이 왜곡됨

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
    difficulty    : 머미나우 전용 이산형(1~4)
    horror        : 머더로그 전용. 0~5, 높을수록 공포 강함
    scene_category: 머미나우 전용 소프트 부스트

변경사항:
    - FAISS 로컬 파일 로드 → Pinecone 인덱스 연결로 교체
    - _dense_search_source(): index.search() → pinecone index.query()로 교체
    - get_embedding(): index.reconstruct() → get_query_embedding()으로 교체
    - BM25용 메타데이터는 Pinecone에서 _fetch_all_metadata()로 로드
====================================================================
"""

import re
import math
import numpy as np
from rank_bm25 import BM25Okapi

from recommender.rag.embeddings import load_index, get_query_embedding

# =========================================================
# 데이터 로드 — Pinecone 연결
# =========================================================
_mm_index, _mm_namespace = load_index("murdermysterylog")  # 머더미스터리로그
_mn_index, _mn_namespace = load_index("murmynow")          # 머미나우


def _fetch_all_metadata(index, namespace: str, source_label: str) -> list[dict]:
    """
    Pinecone 인덱스에서 전체 메타데이터를 페이지 단위로 가져온다.
    BM25 corpus 구성에 사용.
    """
    items = []
    for page in index.list(namespace=namespace):
        # page는 ListResponse 객체 → ID 리스트 추출
        ids = [item.id for item in page.vectors]
        if not ids:
            continue
        fetched = index.fetch(ids=ids, namespace=namespace)
        for _, vec_data in fetched.vectors.items():
            meta = dict(vec_data.metadata or {})
            meta["source"] = source_label
            if "name" in meta and "title" not in meta:
                meta["title"] = meta["name"]
            items.append(meta)
    return items


print("[mm_retriever] Pinecone에서 메타데이터 로드 중...")
_mm_items = _fetch_all_metadata(_mm_index, _mm_namespace, "murdermysterylog")
_mn_items = _fetch_all_metadata(_mn_index, _mn_namespace, "murmynow")

# stats 전용 항목 — BM25 corpus 및 검색 결과에 사용
# review 항목은 min/max_players 등 필터 필드가 없어 품질 저하 유발
_mn_stats_items = [item for item in _mn_items if item.get("type", "stats") != "review"]

print(f"[mm_retriever] 머더미스터리로그: {len(_mm_items)}개")
print(
    f"[mm_retriever] 머미나우 전체: {len(_mn_items)}개 / "
    f"stats 전용: {len(_mn_stats_items)}개"
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
        parts.append(str(item["reviews"]))

    for tag in item.get("emotion_tags", []):
        parts.append(str(tag))

    return " ".join(parts)


# 머더로그 BM25
_mm_corpus    = [_make_searchable_text(s) for s in _mm_items]
_mm_tokenized = [c.lower().split() for c in _mm_corpus]
_mm_bm25      = BM25Okapi(_mm_tokenized)

# 머미나우 BM25 (stats 전용 — review 제외해 IDF 왜곡 방지)
_mn_corpus    = [_make_searchable_text(s) for s in _mn_stats_items]
_mn_tokenized = [c.lower().split() for c in _mn_corpus]
_mn_bm25      = BM25Okapi(_mn_tokenized)

print(
    f"[mm_retriever] BM25 준비 완료 — "
    f"머더로그: {len(_mm_corpus)}개, 머미나우(stats): {len(_mn_corpus)}개"
)


# =========================================================
# 메타데이터 정규화 유틸
# =========================================================
def _as_number(value, default=None):
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
    """True = 통과."""
    # review 항목 제거 (필터/점수 필드 없어 결과 품질 저하)
    if item.get("type") == "review":
        return False

    # 인원 필터
    players = _as_int(query_filter.get("players"), default=None)
    if players is not None:
        max_p = _as_int(item.get("max_players"), default=999)
        min_p = _as_int(item.get("min_players"), default=0)
        if players > max_p or players < min_p:
            return False

    # 시간 필터
    max_time = _as_int(query_filter.get("max_time"), default=None)
    if max_time is not None:
        source = item.get("source", "")
        if source == "murmynow":
            item_time = _as_int(item.get("max_time"), default=None)
        else:
            item_time = _as_int(item.get("play_time"), default=None)
            if item_time is None:
                item_time = _as_int(item.get("max_time"), default=None)
        if item_time is not None and item_time > 0 and item_time > max_time:
            return False

    # scene_category 필터 (머미나우에만 실데이터 존재)
    scene_category = query_filter.get("scene_category")
    if scene_category:
        item_scene = item.get("scene_category") or item.get("유형") or ""
        if item_scene and not _contains(item_scene, scene_category):
            return False

    return True


# =========================================================
# 메타데이터 가중치
# =========================================================
def _metadata_weight(item: dict, query_filter: dict) -> float:
    score = 0.0
    source = item.get("source", "murdermysterylog")

    # 1. rating (0~5, 두 소스 동일 스케일)
    rating = _as_number(item.get("rating"), default=None)
    if rating is not None and rating > 0:
        score += min(rating / 5.0, 1.0) * 12.0

    # 2. difficulty (머미나우 전용 이산형 1~4)
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

    # 3. horror (머더로그 전용, 0~5)
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

    # 4. scene_category 소프트 부스트 (머미나우 전용)
    if query_filter.get("scene_category"):
        item_scene = item.get("scene_category") or item.get("유형") or ""
        if _contains(item_scene, query_filter["scene_category"]):
            score += 5.0

    # 5. 추천 인원 정확도 보너스
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

    # 6. reviews 텍스트 감성/양 보정 (머더로그 전용)
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
    index,
    namespace: str,
    source_label: str,
    query_vector: np.ndarray,
    query_filter: dict,
    topk: int = 200,
) -> dict:
    """
    단일 소스 Pinecone Dense 검색.
    변경 전: faiss index.search() 사용
    변경 후: pinecone index.query() 사용
    반환: {title: {item, rank, score}}
    """
    response = index.query(
        vector=query_vector.tolist()[0],
        top_k=topk * 3,
        namespace=namespace,
        include_metadata=True,
    )
    results: dict = {}
    rank = 0
    for match in response.matches:
        item = dict(match.metadata or {})
        item["source"] = source_label
        if "name" in item and "title" not in item:
            item["title"] = item["name"]
        if not hard_filter(item, query_filter):
            continue
        title = item.get("title") or item.get("name", "")
        if title not in results:
            rank += 1
            results[title] = {
                "item": item,
                "rank": rank,
                "score": float(match.score),
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
    """단일 소스 내 BM25 + Dense RRF 융합 + 메타데이터 가중치."""
    all_titles = set(list(bm25_results.keys()) + list(dense_results.keys()))
    scored = []

    for title in all_titles:
        bm25_data  = bm25_results.get(title)
        dense_data = dense_results.get(title)
        bm25_rank  = bm25_data["rank"]  if bm25_data  else 999
        dense_rank = dense_data["rank"] if dense_data else 999

        if bm25_rank == 999 and dense_rank == 999:
            continue

        rrf_score = 1 / (k + bm25_rank) + 1 / (k + dense_rank)
        if dense_rank == 999:
            rrf_score *= 0.7  # dense 미등장 패널티

        item = (bm25_data or dense_data)["item"]
        meta_score  = _metadata_weight(item, query_filter)
        total_score = rrf_score * scale + meta_score

        item_copy = item.copy()
        item_copy["rrf_score"]   = round(rrf_score, 6)
        item_copy["meta_score"]  = round(meta_score, 2)
        item_copy["total_score"] = round(total_score, 2)
        item_copy["bm25_rank"]   = bm25_rank
        item_copy["dense_rank"]  = dense_rank
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
    max 방식으로 각 아이템이 가장 잘 맞는 소스에서의 순위로만 경쟁.
    """
    mm_rank_map = {
        (item.get("title") or item.get("name", "")): (rank + 1, item)
        for rank, item in enumerate(mm_results)
    }
    mn_rank_map = {
        (item.get("title") or item.get("name", "")): (rank + 1, item)
        for rank, item in enumerate(mn_results)
    }

    all_titles = set(list(mm_rank_map.keys()) + list(mn_rank_map.keys()))
    scored = []

    for title in all_titles:
        mm_entry = mm_rank_map.get(title)
        mn_entry = mn_rank_map.get(title)
        mm_rank  = mm_entry[0] if mm_entry else 999
        mn_rank  = mn_entry[0] if mn_entry else 999

        cross_rrf = max(1 / (k + mm_rank), 1 / (k + mn_rank))

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
    타이틀 리스트 → 평균 임베딩 벡터 반환 (shape: (1, dim)).
    변경 전: faiss index.reconstruct()로 저장된 벡터 직접 추출
    변경 후: get_query_embedding()으로 텍스트를 임베딩해 사용
    """
    if not titles:
        return get_query_embedding("머더미스터리", engine="openai")

    vecs = [get_query_embedding(title, engine="openai") for title in titles]
    return np.mean(vecs, axis=0).reshape(1, -1).astype(np.float32)


# =========================================================
# 공개 인터페이스
# =========================================================
def retrieve(
    query_text: str,
    query_filter: dict,
    query_vector: np.ndarray,
    topk: int = 50,
) -> list:
    """소스별 RRF 후 크로스-소스 재융합 (주력 검색)."""
    intermediate_topk = max(topk * 2, 100)

    # 머더로그
    mm_bm25  = _bm25_search_source(_mm_items, _mm_bm25, query_text, query_filter, topk=200)
    mm_dense = _dense_search_source(_mm_index, _mm_namespace, "murdermysterylog", query_vector, query_filter, topk=200)
    mm_results = _rrf_fuse_source(mm_bm25, mm_dense, query_filter, topk=intermediate_topk)

    # 머미나우 (BM25: stats corpus, Dense: Pinecone 전체)
    mn_bm25  = _bm25_search_source(_mn_stats_items, _mn_bm25, query_text, query_filter, topk=200)
    mn_dense = _dense_search_source(_mn_index, _mn_namespace, "murmynow", query_vector, query_filter, topk=200)
    mn_results = _rrf_fuse_source(mn_bm25, mn_dense, query_filter, topk=intermediate_topk)

    return _cross_source_rrf(mm_results, mn_results, topk=topk)


def retrieve_bm25(
    query_text: str,
    query_filter: dict,
    topk: int = 50,
) -> list:
    """BM25 단독 검색 (두 소스 결과 라운드로빈 인터리브)."""
    mm_res = _bm25_search_source(_mm_items, _mm_bm25, query_text, query_filter, topk=topk)
    mn_res = _bm25_search_source(_mn_stats_items, _mn_bm25, query_text, query_filter, topk=topk)
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
    mm_res = _dense_search_source(_mm_index, _mm_namespace, "murdermysterylog", query_vector, query_filter, topk=topk)
    mn_res = _dense_search_source(_mn_index, _mn_namespace, "murmynow", query_vector, query_filter, topk=topk)
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
