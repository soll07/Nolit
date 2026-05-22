"""
yoonha_hybrid_retriever.py
category 기반 라우터 — yoonha_graph.py 에서 이 파일만 호출하면 됨.

====================================================================
[역할]
    그래프(yoonha_graph.py)와 개별 retriever 사이의 중간 계층.
    category 값("boardgame" | "murdermystery" | "escape")에 따라
    적절한 retriever 모듈로 요청을 라우팅한다.

[설계 의도]
    - 그래프 코드가 개별 retriever를 직접 알 필요 없이
      이 파일의 retrieve() 하나만 호출하면 된다.
    - 새로운 카테고리 추가 시 이 파일에만 라우팅 분기를 추가하면
      그래프 코드는 수정 불필요.

[내부 라우팅]
    "boardgame"     → yoonha_boardgame_retriever.retrieve()
    "murdermystery" → yoonha_mm_retriever.retrieve()
    "escape"        → jihye_bbabang_retriever.retrieve()

[제공 함수 — 4종 검색 + 1종 임베딩]
    retrieve()         : BM25 + FAISS RRF 하이브리드 검색 (주력)
    retrieve_bm25()    : BM25 단독 (디버깅/평가용)
    retrieve_dense()   : FAISS 단독 (디버깅/평가용)
    retrieve_vanilla() : 필터+평점순 (FAISS 장애 시 fallback)
    get_embedding()    : 앵커 타이틀 → 임베딩 벡터 변환
====================================================================
"""

import numpy as np
from pathlib import Path
import sys

_RAG_DIR = str(Path(__file__).resolve().parent)
if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

# ← 상단 import 전부 제거


def retrieve(query_text, query_filter, query_vector, category, topk=50):
    if category == "boardgame":
        from recommender.rag.yoonha_boardgame_retriever import retrieve as _fn
        return _fn(query_text, query_filter, query_vector, topk)
    elif category == "murdermystery":
        from recommender.rag.yoonha_mm_retriever import retrieve as _fn
        return _fn(query_text, query_filter, query_vector, topk)
    elif category == "escape":
        from recommender.rag.jihye_bbabang_retriever import retrieve as _fn
        return _fn(query_text, query_filter, query_vector, topk)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}")


def retrieve_bm25(query_text, query_filter, category, topk=50):
    if category == "boardgame":
        from recommender.rag.yoonha_boardgame_retriever import retrieve_bm25 as _fn
        return _fn(query_text, query_filter, topk)
    elif category == "murdermystery":
        from recommender.rag.yoonha_mm_retriever import retrieve_bm25 as _fn
        return _fn(query_text, query_filter, topk)
    elif category == "escape":
        from recommender.rag.jihye_bbabang_retriever import retrieve_bm25 as _fn
        return _fn(query_text, query_filter, topk)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}")


def retrieve_dense(query_vector, query_filter, category, topk=50):
    if category == "boardgame":
        from recommender.rag.yoonha_boardgame_retriever import retrieve_dense as _fn
        return _fn(query_vector, query_filter, topk)
    elif category == "murdermystery":
        from recommender.rag.yoonha_mm_retriever import retrieve_dense as _fn
        return _fn(query_vector, query_filter, topk)
    elif category == "escape":
        from recommender.rag.jihye_bbabang_retriever import retrieve_dense as _fn
        return _fn(query_vector, query_filter, topk)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}")


def retrieve_vanilla(query_filter, category, topk=50):
    if category == "boardgame":
        from recommender.rag.yoonha_boardgame_retriever import retrieve_vanilla as _fn
        return _fn(query_filter, topk)
    elif category == "murdermystery":
        from recommender.rag.yoonha_mm_retriever import retrieve_vanilla as _fn
        return _fn(query_filter, topk)
    elif category == "escape":
        from recommender.rag.jihye_bbabang_retriever import retrieve_vanilla as _fn
        return _fn(query_filter, topk)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}")


def get_embedding(titles, category):
    if category == "boardgame":
        from recommender.rag.yoonha_boardgame_retriever import get_embedding as _fn
        return _fn(titles)
    elif category == "murdermystery":
        from recommender.rag.yoonha_mm_retriever import get_embedding as _fn
        return _fn(titles)
    elif category == "escape":
        from recommender.rag.jihye_bbabang_retriever import get_embedding as _fn
        return _fn(titles)
    else:
        raise ValueError(f"알 수 없는 category: {category!r}")