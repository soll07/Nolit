"""
recommender/rag/yoonha_boardgame_retriever.py

[ 역할 ]
  BGG + 보드라이프 통합 BM25 + Pinecone RRF 하이브리드 검색

[ 검색 흐름 ]
  1. BM25 키워드 검색 → 키워드 매칭 기반 랭킹
  2. Pinecone dense 검색 → 의미 유사도 기반 랭킹
  3. RRF(Reciprocal Rank Fusion) → 두 랭킹을 결합
  4. 메타데이터 가중치 → 평점/카테고리/인원 등 보정
  5. 최종 total_score 계산 후 정렬

[ RRF 공식 ]
  rrf_score = 1/(k + bm25_rank) + 1/(k + dense_rank)
  - k=60 (기본값): 높은 랭크에 집중하면서도 중간 랭크도 반영
  - BM25에만 있는 아이템: rrf_score × 0.7 (dense 미등장 패널티)

[ 최종 점수 ]
  total_score = rrf_score × 3000 + metadata_weight
  - 3000 곱하기: rrf_score(0.001~0.03)를 metadata_weight(0~30)와 스케일 맞춤

[ 데이터 소스별 특이사항 ]
  - BGG: 10점 만점, playing_time 단일값, weight(복잡도) 0~5
  - 보드라이프: 5점 만점, min_time/max_time 범위형, best_players 존재
  - 보드라이프 가중치 1.5배 (한국 서비스이므로 한국 데이터 우선)

변경사항:
  - FAISS 로컬 파일 로드 → Pinecone 인덱스 연결로 교체
  - _dense_search(): index.search() → pinecone index.query()로 교체
  - get_embedding(): index.reconstruct() → get_query_embedding()으로 교체
  - BM25용 메타데이터는 Pinecone에서 가져온 metadata 필드 사용
  - reviews meta: Pinecone bgg_reviews / boardlife_reviews 네임스페이스에서 집계
"""

import json
import math
import re
import statistics
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

from recommender.rag.embeddings import load_index, get_query_embedding

# -------------------------
# 데이터 로드 — Pinecone 연결
# -------------------------
# Pinecone 인덱스 객체 + 네임스페이스
bgg_index,  bgg_namespace  = load_index("bgg_stats")
bl_index,   bl_namespace   = load_index("boardlife_stats")
bgg_rev_index, bgg_rev_namespace = load_index("bgg_reviews")
bl_rev_index,  bl_rev_namespace  = load_index("boardlife_reviews")

# -------------------------
# BM25용 메타데이터 로드
# Pinecone에서 전체 메타데이터를 가져와 BM25 corpus 구성
# -------------------------
def _fetch_all_metadata(index, namespace: str, source_label: str) -> list[dict]:
    items = []
    
    # list() 대신 query()로 데이터 가져오기
    # 더미 벡터(0으로 채운 벡터)로 쿼리해서 전체 메타데이터 조회
    index_info = index.describe_index_stats()
    dim = index_info.dimension  # 인덱스 차원수 확인
    
    dummy_vector = [0.0] * dim
    response = index.query(
        vector=dummy_vector,
        top_k=10000,           # 최대한 많이 가져오기
        namespace=namespace,
        include_metadata=True,
    )
    
    for match in response.matches:
        meta = dict(match.metadata or {})
        meta["_pinecone_id"] = match.id
        meta["source"] = source_label
        items.append(meta)
    
    return items
    # """
    # Pinecone 인덱스에서 전체 메타데이터를 페이지 단위로 가져온다.
    # BM25 corpus 구성 및 리뷰 집계에 사용.
    # """
    # items = []
    # # Pinecone list() API로 모든 벡터 ID 조회 후 fetch
    # for id_batch in index.list(namespace=namespace):
    #     fetched = index.fetch(ids=id_batch, namespace=namespace)
    #     for vec_id, vec_data in fetched.vectors.items():
    #         meta = dict(vec_data.metadata or {})
    #         meta["_pinecone_id"] = vec_id
    #         meta["source"] = source_label
    #         items.append(meta)
    # return items


print("[boardgame_retriever] Pinecone에서 메타데이터 로드 중...")
bgg_stats = _fetch_all_metadata(bgg_index, bgg_namespace, "bgg")
bl_stats  = _fetch_all_metadata(bl_index,  bl_namespace,  "boardlife")
all_items = bgg_stats + bl_stats
print(f"[boardgame_retriever] BGG: {len(bgg_stats)}개, 보드라이프: {len(bl_stats)}개")

# -------------------------
# reviews meta 집계 — Pinecone reviews 네임스페이스에서
# -------------------------
def _build_review_map(index, namespace: str) -> dict[str, dict]:
    """
    Pinecone reviews 네임스페이스 → title별 review 통계 dict 빌드.
    {
      "Brass: Birmingham": {
          "review_count": 4557,
          "review_avg":   8.344,
          "review_stdev": 1.810,
      }, ...
    }
    """
    raw: dict[str, list[float]] = {}
    for page in index.list(namespace=namespace):
        ids = [item.id for item in page.vectors]
        if not ids:
            continue
        fetched = index.fetch(ids=ids, namespace=namespace)
        for _, vec_data in fetched.vectors.items():
            meta = vec_data.metadata or {}
            r = meta.get("rating")
            title = meta.get("title")
            if not title or r is None or r == "":
                continue
            try:
                raw.setdefault(title, []).append(float(r))
            except (ValueError, TypeError):
                continue

    result = {}
    for title, ratings in raw.items():
        result[title] = {
            "review_count": len(ratings),
            "review_avg":   round(sum(ratings) / len(ratings), 3),
            "review_stdev": round(statistics.stdev(ratings), 3) if len(ratings) > 1 else 0.0,
        }
    return result
    # raw: dict[str, list[float]] = {}
    # for id_batch in index.list(namespace=namespace):
    #     fetched = index.fetch(ids=id_batch, namespace=namespace)
    #     for _, vec_data in fetched.vectors.items():
    #         meta = vec_data.metadata or {}
    #         r = meta.get("rating")
    #         title = meta.get("title")
    #         if r is None or title is None:
    #             continue
    #         raw.setdefault(title, []).append(float(r))

    # result = {}
    # for title, ratings in raw.items():
    #     result[title] = {
    #         "review_count": len(ratings),
    #         "review_avg":   round(sum(ratings) / len(ratings), 3),
    #         "review_stdev": round(statistics.stdev(ratings), 3) if len(ratings) > 1 else 0.0,
    #     }
    # return result


print("[boardgame_retriever] 리뷰 통계 집계 중...")
_bgg_review_map = _build_review_map(bgg_rev_index, bgg_rev_namespace)
_bl_review_map  = _build_review_map(bl_rev_index,  bl_rev_namespace)
print(f"[boardgame_retriever] BGG 리뷰 집계: {len(_bgg_review_map)}개 게임")
print(f"[boardgame_retriever] 보드라이프 리뷰 집계: {len(_bl_review_map)}개 게임")


def _get_review_data(item: dict) -> dict | None:
    """아이템 소스에 맞는 review 집계 데이터 반환."""
    title = item.get("title", "")
    if item.get("source") == "bgg":
        return _bgg_review_map.get(title)
    else:
        return _bl_review_map.get(title) or _bgg_review_map.get(item.get("title_eng", ""))


# -------------------------
# BM25 준비
# -------------------------
CATEGORY_KO = {
    "Strategy": "전략", "Economic": "경제", "Party": "파티", "War": "전쟁",
    "Family": "가족", "Abstract": "추상", "Thematic": "테마", "Adventure": "어드벤처",
    "Fantasy": "판타지", "Horror": "공포", "Science Fiction": "SF",
    "Deduction": "추리", "Negotiation": "협상", "Cooperative": "협력",
    "Card Game": "카드게임", "Dice": "주사위", "Puzzle": "퍼즐",
}
MECHANISM_KO = {
    "Worker Placement": "일꾼배치", "Deck Building": "덱빌딩", "Engine Building": "엔진빌딩",
    "Area Control": "지역장악", "Cooperative Game": "협력", "Auction": "경매",
    "Market": "시장", "Hand Management": "패관리", "Tile Placement": "타일배치",
    "Route Building": "루트빌딩", "Push Your Luck": "운빨", "Voting": "투표",
    "Drafting": "드래프팅", "Roll and Write": "롤앤라이트",
}
BM25_STOPWORDS = {
    "one","two","three","four","five","six","seven","eight","nine","ten",
    "a","an","the","of","in","at","to","for","and","or","is","it",
    "city","cities","world","age","game","games","edition","player","players"
}


def _translate_tags(tags_raw, mapping):
    if isinstance(tags_raw, list):
        tags = tags_raw
    elif isinstance(tags_raw, str) and tags_raw:
        tags = tags_raw.split("|")
    else:
        return []
    result = list(tags)
    for tag in tags:
        ko = mapping.get(tag.strip())
        if ko:
            result.append(ko)
    return result


def _make_searchable_text(item):
    parts = [str(item.get("title", ""))]
    if item.get("title_eng"):
        parts.append(str(item["title_eng"]))
    parts.extend(_translate_tags(item.get("category", ""), CATEGORY_KO))
    parts.extend(_translate_tags(item.get("mechanism", ""), MECHANISM_KO))
    t = item.get("type", "")
    if isinstance(t, list):
        parts.extend(t)
    elif isinstance(t, str) and t:
        parts.append(t)
    des = item.get("designer", "")
    if isinstance(des, list):
        parts.extend(des)
    elif isinstance(des, str) and des:
        parts.extend(des.split("|"))
    for tag in item.get("emotion_tags", []):
        parts.append(str(tag))
    return " ".join([str(p) for p in parts if p])


_corpus = [_make_searchable_text(s) for s in all_items]
_tokenized_corpus = [c.lower().split() for c in _corpus]
_bm25 = BM25Okapi(_tokenized_corpus)
print(f"[boardgame_retriever] BM25 corpus 준비 완료: {len(_corpus)}개")


# -------------------------
# 메타데이터 정규화 유틸
# -------------------------
def _as_number(value, default=None):
    """문자열/숫자 메타데이터를 float로 안전 변환. None/NaN/inf는 default로 처리."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        try:
            if math.isnan(value) or math.isinf(value):
                return default
        except TypeError:
            pass
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned or cleaned.lower() in {"none", "null", "nan", "na", "n/a", "?", "-"}:
            return default
        m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if m:
            try:
                num = float(m.group(0))
                if math.isnan(num) or math.isinf(num):
                    return default
                return num
            except ValueError:
                return default
    return default


def _as_int(value, default=None):
    num = _as_number(value, default=None)
    if num is None:
        return default
    try:
        if math.isnan(num) or math.isinf(num):
            return default
    except TypeError:
        return default
    return int(num)


def _split_tags(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [t.strip() for t in re.split(r"[|,/]", value) if t.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _contains_tag(value, target: str) -> bool:
    if not target:
        return False
    target_lower = target.lower()
    tags = _split_tags(value)
    expanded = []
    expanded.extend(tags)
    expanded.extend(_translate_tags(tags, CATEGORY_KO))
    expanded.extend(_translate_tags(tags, MECHANISM_KO))
    return any(target_lower in t.lower() or t.lower() in target_lower for t in expanded)


def _parse_player_values(value) -> set[int]:
    values: set[int] = set()
    if value is None:
        return values
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return values
        values.add(int(value))
        return values
    if isinstance(value, int):
        values.add(value)
        return values
    if isinstance(value, list):
        for v in value:
            values |= _parse_player_values(v)
        return values
    if isinstance(value, dict):
        for v in value.values():
            values |= _parse_player_values(v)
        return values
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan", "na", "n/a", "-", "?"}:
        return values
    for m in re.findall(r"\d+", text):
        try:
            values.add(int(m))
        except ValueError:
            continue
    return values


def _normalized_rating(item: dict) -> float | None:
    rating = _as_number(item.get("avg_rating"), default=None)
    if rating is None or rating <= 0:
        return None
    source = item.get("source")
    if source == "bgg":
        return min(rating / 10.0, 1.0)
    if source == "boardlife":
        return min(rating / 5.0, 1.0)
    return None


def _source_weight(item: dict, query_filter: dict) -> float:
    pref = query_filter.get("source_pref") or query_filter.get("source_preference") or "korean"
    if pref in {"global", "bgg"}:
        return 1.0
    if item.get("source") == "boardlife":
        return 1.5
    return 1.0


# -------------------------
# 하드 필터
# -------------------------
def hard_filter(item: dict, query_filter: dict) -> bool:
    """조건 불만족 아이템 제거. True = 통과."""
    players = _as_int(query_filter.get("players"), default=None)
    if players is not None:
        max_p = _as_int(item.get("max_players"), default=999)
        min_p = _as_int(item.get("min_players"), default=0)
        if players > max_p or players < min_p:
            return False

    max_time = _as_int(query_filter.get("playing_time"), default=None)
    if max_time is not None:
        if item.get("source") == "boardlife":
            item_time = _as_int(item.get("max_time"), default=None)
        else:
            item_time = _as_int(item.get("playing_time"), default=None)
        if item_time is not None and item_time > 0 and item_time > max_time:
            return False

    if query_filter.get("strict_weight_filter"):
        w = _as_number(item.get("weight"), default=None)
        if w is not None:
            weight_max = _as_number(query_filter.get("weight_max"), default=None)
            if weight_max is not None and w > weight_max:
                return False
            if query_filter.get("weight_pref") == "heavy" and w < 3.5:
                return False

    return True


# -------------------------
# 메타데이터 가중치
# -------------------------
def _metadata_weight(item: dict, query_filter: dict) -> float:
    score = 0.0
    source = item.get("source", "bgg")

    normalized = _normalized_rating(item)
    if normalized is not None:
        review_data = _get_review_data(item)
        if review_data:
            count = _as_number(review_data.get("review_count"), default=0) or 0
            stdev = _as_number(review_data.get("review_stdev"), default=0) or 0
            confidence_base = 500.0 if source == "bgg" else 50.0
            confidence = min(count / confidence_base, 1.0)
            stdev_penalty = min(stdev / 4.5, 1.0) * 0.2
            score += normalized * 15.0 * (0.6 + 0.4 * confidence) * (1.0 - stdev_penalty)
        else:
            score += normalized * 15.0 * 0.6

    req_category = query_filter.get("category")
    if req_category:
        item_category = item.get("category")
        if _contains_tag(item_category, req_category):
            score += 10.0
        elif item_category and _split_tags(item_category):
            score -= 10.0
        else:
            score -= 3.0

    req_mechanism = query_filter.get("mechanism")
    if req_mechanism:
        item_mechanism = item.get("mechanism")
        if _contains_tag(item_mechanism, req_mechanism):
            score += 8.0
        elif item_mechanism and _split_tags(item_mechanism):
            score -= 8.0
        else:
            score -= 2.0

    players = _as_int(query_filter.get("players"), default=None)
    if players is not None:
        if source == "bgg":
            rec_values = _parse_player_values(item.get("recommended_players"))
        else:
            rec_values = (
                _parse_player_values(item.get("best_players"))
                | _parse_player_values(item.get("recommend_players"))
                | _parse_player_values(item.get("recommended_players"))
            )
        if players in rec_values:
            score += 6.0

    pref = query_filter.get("weight_pref")
    w = _as_number(item.get("weight"), default=None)
    if pref and w is not None:
        if pref == "light":
            score += max(0.0, (3.0 - w) / 3.0) * 6.0
            if w >= 3.0:
                score -= min((w - 2.5) * 4.0, 10.0)
        elif pref == "medium":
            score += max(0.0, 1.0 - abs(w - 3.0) / 2.0) * 6.0
        elif pref == "heavy":
            score += max(0.0, (w - 2.5) / 2.5) * 6.0
            if w <= 2.5:
                score -= min((3.0 - w) * 4.0, 10.0)

    cr = item.get("category_rank")
    overall = None
    if isinstance(cr, dict):
        overall = _as_number(cr.get("Overall"), default=None)
    elif isinstance(cr, str):
        try:
            cr_dict = json.loads(cr.replace("'", '"'))
            overall = _as_number(
                cr_dict.get("Overall") or cr_dict.get("전략") or cr_dict.get("가족"),
                default=None,
            )
        except Exception:
            overall = _as_number(cr, default=None)
    if overall is not None and overall > 0:
        score += max(0.0, 10.0 - math.log1p(overall) * 1.35)

    rank = _as_number(item.get("rank"), default=None)
    if rank is not None and rank > 0:
        score += max(0.0, 5.0 - math.log1p(rank) * 0.75)

    review_data = _get_review_data(item)
    if review_data:
        count = _as_number(review_data.get("review_count"), default=0) or 0
        base = 500.0 if source == "bgg" else 50.0
        score += min(math.log1p(count) / math.log1p(base), 1.0) * 3.0

    return score


# -------------------------
# 검색 내부 함수
# -------------------------
def _bm25_search(query_text: str, query_filter: dict, topk: int = 200) -> dict:
    """BM25 키워드 검색 — 기존과 동일 (메모리 내 all_items 사용)."""
    tokens = [t for t in query_text.lower().split() if t not in BM25_STOPWORDS]
    scores = _bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1]
    results = {}
    rank = 0
    for idx in top_idx:
        if idx >= len(all_items):
            continue
        item = all_items[idx]
        if not hard_filter(item, query_filter):
            continue
        key = f"{item['source']}::{item.get('title', '')}"
        if key not in results:
            rank += 1
            results[key] = {"item": item, "rank": rank, "bm25_score": float(scores[idx])}
            if rank >= topk:
                break
    return results


def _dense_search(query_vector: np.ndarray, query_filter: dict, topk: int = 200) -> dict:
    """
    Pinecone dense 검색 — BGG + 보드라이프 각각 query 후 score 기준 합산 정렬.
    변경 전: faiss index.search() 사용
    변경 후: pinecone index.query() 사용
    """
    results = {}

    for index, namespace, meta_list in [
        (bgg_index, bgg_namespace, bgg_stats),
        (bl_index,  bl_namespace,  bl_stats),
    ]:
        response = index.query(
            vector=query_vector.tolist()[0],
            top_k=topk * 3,
            namespace=namespace,
            include_metadata=True,
        )
        for match in response.matches:
            item = dict(match.metadata or {})
            item["source"] = "bgg" if namespace == bgg_namespace else "boardlife"
            if not hard_filter(item, query_filter):
                continue
            key = f"{item['source']}::{item.get('title', '')}"
            if key not in results:
                # Pinecone은 유사도 score(높을수록 좋음) 반환 → l2_dist 대신 사용
                results[key] = {"item": item, "score": float(match.score)}

    # score 기준 내림차순 정렬 후 rank 부여
    sorted_items = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
    ranked = {}
    for rank, (key, data) in enumerate(sorted_items[:topk], 1):
        ranked[key] = {**data, "rank": rank}
    return ranked


def _rrf_fuse(
    bm25_results: dict,
    dense_results: dict,
    query_filter: dict,
    topk: int,
    k: int = 60,
) -> list[dict]:
    """RRF 융합 + 메타 가중치 적용."""
    all_keys = set(list(bm25_results.keys()) + list(dense_results.keys()))
    scored = []
    for key in all_keys:
        bm25_data  = bm25_results.get(key)
        dense_data = dense_results.get(key)
        bm25_rank  = bm25_data["rank"]  if bm25_data  else 999
        dense_rank = dense_data["rank"] if dense_data else 999
        if bm25_rank == 999 and dense_rank == 999:
            continue

        rrf_score = 1 / (k + bm25_rank) + 1 / (k + dense_rank)

        if dense_rank == 999:
            rrf_score *= 0.7

        item = (bm25_data or dense_data)["item"]
        source_weight = _source_weight(item, query_filter)
        rrf_weighted  = rrf_score * source_weight
        meta_score    = _metadata_weight(item, query_filter)
        total_score   = rrf_weighted * 3000 + meta_score

        item_copy = item.copy()
        item_copy["rrf_score"]     = round(rrf_score, 6)
        item_copy["meta_score"]    = round(meta_score, 2)
        item_copy["total_score"]   = round(total_score, 2)
        item_copy["bm25_rank"]     = bm25_rank
        item_copy["dense_rank"]    = dense_rank
        item_copy["source_weight"] = source_weight
        scored.append(item_copy)

    scored.sort(key=lambda x: x["total_score"], reverse=True)
    scored = _deduplicate_editions(scored)
    return scored[:topk]


def _normalize_title_for_dedup(title: str) -> str:
    if not title:
        return ""
    normalized = title.lower().strip()
    normalized = re.sub(r"\s*\([^)]*edition[^)]*\)", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\s*[:：\-–—]\s*(second|third|revised|new|anniversary|deluxe|ultimate)\s*edition.*$",
        "", normalized, flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s*\(\d{4}\)\s*$", "", normalized)
    return normalized.strip()


def _deduplicate_editions(scored: list[dict]) -> list[dict]:
    seen_base_titles: dict[str, bool] = {}
    deduped: list[dict] = []
    for item in scored:
        title = item.get("title", "")
        base = _normalize_title_for_dedup(title)
        dedup_key = f"{item.get('source', '')}::{base}"
        if dedup_key not in seen_base_titles:
            seen_base_titles[dedup_key] = True
            deduped.append(item)
    return deduped


# -------------------------
# 앵커 임베딩 유틸
# -------------------------
def get_embedding(titles: list[str]) -> np.ndarray:
    """
    타이틀 리스트 → 평균 임베딩 벡터 반환 (shape: (1, dim)).
    변경 전: faiss index.reconstruct()로 저장된 벡터 직접 추출
    변경 후: get_query_embedding()으로 텍스트를 임베딩해 사용
    타이틀을 못 찾으면 빈 쿼리 임베딩으로 fallback.
    """
    if not titles:
        return get_query_embedding("보드게임", engine="openai")

    vecs = []
    for title in titles:
        vec = get_query_embedding(title, engine="openai")
        vecs.append(vec)

    return np.mean(vecs, axis=0).reshape(1, -1).astype(np.float32)


# -------------------------
# 공개 인터페이스
# -------------------------
def retrieve(
    query_text: str,
    query_filter: dict,
    query_vector: np.ndarray,
    topk: int = 50,
) -> list[dict]:
    """RRF 하이브리드 검색 (BM25 + Pinecone 융합)."""
    bm25_res  = _bm25_search(query_text, query_filter, topk=200)
    dense_res = _dense_search(query_vector, query_filter, topk=200)
    return _rrf_fuse(bm25_res, dense_res, query_filter, topk=topk)


def retrieve_bm25(
    query_text: str,
    query_filter: dict,
    topk: int = 50,
) -> list[dict]:
    """BM25 단독 검색."""
    results = _bm25_search(query_text, query_filter, topk=topk)
    items = sorted(results.values(), key=lambda x: x["rank"])
    return [d["item"] for d in items]


def retrieve_dense(
    query_vector: np.ndarray,
    query_filter: dict,
    topk: int = 50,
) -> list[dict]:
    """Dense 단독 검색."""
    results = _dense_search(query_vector, query_filter, topk=topk)
    items = sorted(results.values(), key=lambda x: x["rank"])
    return [d["item"] for d in items]


def retrieve_vanilla(
    query_filter: dict,
    topk: int = 50,
) -> list[dict]:
    """Vanilla 검색 — 필터 통과 후 평점 내림차순."""
    results = [item.copy() for item in all_items if hard_filter(item, query_filter)]
    results.sort(key=lambda x: x.get("avg_rating") or 0, reverse=True)
    return results[:topk]
