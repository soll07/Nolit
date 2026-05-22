"""
insert_vectors.py
FAISS .index + _meta.json → Pinecone 삽입 스크립트

[구조]
  .index       : description을 임베딩한 벡터
  _meta.json   : rank, title 등 메타데이터
  CSV          : 사용 안 함 (임베딩할 때만 필요했던 파일)

[인덱스 구조]
  nolit-boardgame  (dim=1536): boardlife_stats, bgg_stats
  nolit-escape     (dim=1536) : bbabang_stats, bbabang_reviews
  nolit-crimescene (dim=1536): murdermysterylog, murmynow

[WU 최적화]
  - 벡터 수가 동일하면 재삽입 스킵 (--force 옵션으로 강제 재삽입 가능)
  - delete는 --force 시에만 실행
  - API 키는 .env 파일에서 로드
"""

import json
import math
import os
import argparse
import hashlib
import faiss
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# ── 환경변수 로드 ──────────────────────────────────────────
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise EnvironmentError(
        "PINECONE_API_KEY가 설정되지 않았습니다.\n"
        ".env 파일에 PINECONE_API_KEY=your_key_here 를 추가해 주세요."
    )

BASE = Path("../../04_vectorstore")

# ── 소스별 설정 ────────────────────────────────────────────
SOURCES = [
    # 보드게임 - boardlife
    (
        "faiss_boardlife_stats.index",
        "faiss_boardlife_stats_meta.json",
        "boardlife_stats",
        "nolit-boardgame",
        "boardlife_stats",
        ["rank", "title", "title_eng", "age", "weight", "designer", "artist",
         "type", "category", "mechanism", "image", "avg_rating", "category_rank",
         "min_players", "max_players", "best_players", "recommend_players",
         "min_time", "max_time"],
    ),
    # 보드게임 - boardlife 리뷰
    (
        "faiss_boardlife_reviews.index",
        "faiss_boardlife_reviews_meta.json",
        "boardlife_reviews",
        "nolit-boardgame",
        "boardgame",
        ["rank", "title", "rating"],
    ),
    # 보드게임 - bgg
    (
        "faiss_bgg_stats.index",
        "faiss_bgg_stats_meta.json",
        "bgg_stats",
        "nolit-boardgame",
        "bgg_stats",
        ["rank", "title", "min_players", "max_players", "recommended_players",
         "playing_time", "age", "weight", "avg_rating", "designer", "artist",
         "type", "category", "mechanism", "awards", "rank_all"],
    ),
    # 방탈출 - bbabang 소개 (dim=768)
    (
        "bbabang_stats.index",
        "bbabang_stats_metadata.json",
        "bbabang_stats",
        "nolit-escape",
        "bbabang_stats",
        ["title", "store_name", "area", "location", "playing_time",
         "max_players", "price", "difficulty", "horror", "activity",
         "satisfaction", "puzzle", "story", "interior", "production"],
    ),
    # 방탈출 - bbabang 리뷰 (dim=768)
    (
        "bbabang_reviews.index",
        "bbabang_reviews_metadata.json",
        "bbabang_reviews",
        "nolit-escape",
        "bbabang_reviews",
        ["title", "store_name", "chunk_index", "document"],
    ),
    # 머더미스터리 - 머더미스터리로그
    (
        "faiss_murdermysterylog.index",
        "faiss_murdermysterylog_meta.json",
        "murdermysterylog",
        "nolit-crimescene",
        "murdermysterylog",
        ["url", "name", "rating", "play_time", "description",
         "시리즈", "제작", "min_players", "max_players", "source"],
    ),
    # 머더미스터리 - 머미나우
    (
        "faiss_murmynow.index",
        "faiss_murmynow_meta.json",
        "murmynow",
        "nolit-crimescene",
        "murmynow",
        ["ref_id", "name", "rating", "play_time", "min_players", "max_players",
         "review_text", "source"],
    ),
]


# ── 유틸 ──────────────────────────────────────────────────
def clean_val(val):
    """Pinecone 메타데이터 값 정제
    - None, NaN → 빈 문자열
    - list → " | " 로 합친 문자열
    - 나머지는 그대로
    """
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    if isinstance(val, list):
        return " | ".join(str(v) for v in val)
    return val


def load_meta(filename: str) -> list:
    """JSON 메타데이터 파일 로드"""
    path = BASE / filename
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_metadata(row: dict, source: str, meta_cols: list) -> dict:
    """Pinecone에 저장할 메타데이터 딕셔너리 생성"""
    meta = {"source": source}
    for col in meta_cols:
        meta[col] = clean_val(row.get(col))
    return meta


def get_meta_hash(meta_file: str) -> str:
    """메타데이터 파일의 MD5 해시 반환 (변경 감지용)"""
    path = BASE / meta_file
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_index_hash(index_file: str) -> str:
    """FAISS 인덱스 파일의 MD5 해시 반환 (변경 감지용)"""
    path = BASE / index_file
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_hash_cache(cache_path: Path) -> dict:
    """이전 실행의 해시 캐시 로드"""
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_hash_cache(cache_path: Path, cache: dict):
    """현재 해시를 캐시 파일에 저장"""
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def init_indexes(pc: Pinecone):
    """Pinecone 인덱스 없으면 생성"""
    for name, dim in [
        ("nolit-boardgame",      1536),
        ("nolit-escape",  768),
        ("nolit-crimescene",     1536),
    ]:
        if name not in pc.list_indexes().names():
            pc.create_index(
                name=name,
                dimension=dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"  ✅ {name} 인덱스 생성 완료")
        else:
            print(f"  ✔  {name} 인덱스 이미 존재")


def insert_source(
    pc, index_file, meta_file, source, index_name, namespace, meta_cols,
    hash_cache: dict, force: bool = False, batch_size: int = 100
):
    """FAISS 벡터 + JSON 메타데이터 → Pinecone 삽입
    
    - force=False : 파일 해시가 동일하면 스킵 (WU 절약)
    - force=True  : 항상 delete 후 재삽입
    """
    faiss_path = BASE / index_file
    if not faiss_path.exists():
        print(f"  [SKIP] {index_file} 없음")
        return

    # ── 변경 감지 ──────────────────────────────────────────
    current_hash = get_index_hash(index_file) + get_meta_hash(meta_file)
    cache_key    = source

    if not force and hash_cache.get(cache_key) == current_hash:
        print(f"  [SKIP] {source} — 변경 없음 (WU 절약)")
        return

    # ── FAISS 로드 ─────────────────────────────────────────
    faiss_index = faiss.read_index(str(faiss_path))
    total = faiss_index.ntotal
    print(f"\n  [{source}] 벡터 {total}개 → {index_name} / {namespace}  (dim={faiss_index.d})")

    # ── 메타데이터 로드 ────────────────────────────────────
    try:
        meta_list = load_meta(meta_file)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return

    if len(meta_list) != total:
        print(f"  ❌ 벡터 수({total})와 메타데이터 수({len(meta_list)}) 불일치! 삽입 중단")
        return

    # ── Pinecone 기존 데이터 초기화 (force 시에만) ─────────
    pinecone_index = pc.Index(index_name)
    if force:
        try:
            pinecone_index.delete(delete_all=True, namespace=namespace)
            print(f"  🗑  기존 데이터 초기화 완료 (--force)")
        except Exception:
            pass

    # ── 벡터 + 메타데이터 삽입 ─────────────────────────────
    all_vecs = faiss_index.reconstruct_n(0, total)
    vectors  = []

    for idx in range(total):
        meta = build_metadata(meta_list[idx], source, meta_cols)
        vectors.append({
            "id":       f"{source}_{idx + 1}",
            "values":   all_vecs[idx].tolist(),
            "metadata": meta,
        })

        if len(vectors) == batch_size:
            pinecone_index.upsert(vectors=vectors, namespace=namespace)
            vectors = []

    if vectors:
        pinecone_index.upsert(vectors=vectors, namespace=namespace)

    print(f"  ✅ {total}건 완료")

    # ── 해시 캐시 업데이트 ─────────────────────────────────
    hash_cache[cache_key] = current_hash


def run():
    parser = argparse.ArgumentParser(description="FAISS → Pinecone 삽입 스크립트")
    parser.add_argument(
        "--force", action="store_true",
        help="변경 여부와 관계없이 모든 소스를 강제 재삽입 (delete 후 upsert)"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="특정 소스만 삽입 (예: --source boardlife_stats)"
    )
    args = parser.parse_args()

    pc = Pinecone(api_key=PINECONE_API_KEY)

    print("[인덱스 초기화]")
    init_indexes(pc)

    # 해시 캐시 로드
    cache_path = BASE / ".insert_cache.json"
    hash_cache = load_hash_cache(cache_path)

    print("\n[벡터 + 메타데이터 삽입]")
    for index_file, meta_file, source, index_name, namespace, meta_cols in SOURCES:
        # --source 옵션으로 특정 소스만 선택
        if args.source and source != args.source:
            continue

        insert_source(
            pc, index_file, meta_file, source, index_name, namespace, meta_cols,
            hash_cache=hash_cache,
            force=args.force,
        )

    # 해시 캐시 저장
    save_hash_cache(cache_path, hash_cache)

    print("\n=== 전체 완료 ===")


if __name__ == "__main__":
    run()