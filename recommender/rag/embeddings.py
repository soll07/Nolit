"""
recommender/rag/embeddings.py

[ 역할 ]
  - Pinecone 인덱스 생성 및 저장 전담
  - OpenAI (1536차원) / HuggingFace (768차원) 두 엔진 지원
  - config.yaml에서 설정, loader.py에서 데이터 수신

[ 사용법 ]
  python manage.py embed_contents --all
  python manage.py embed_contents --source bgg_stats
  python manage.py embed_contents --source bbabang_stats
"""

import os
import time
import math
import numpy as np
import openai
import faiss
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

from .config import (
    get_config,
    get_data_dir,
    get_embedding_cfg,
    get_pinecone_cfg,
    get_source_pinecone,
    list_sources,
)
from .loader import load_source


# ══════════════════════════════════════════════
# 0. 초기화
# ══════════════════════════════════════════════

_cfg     = get_config()
_emb_cfg = get_embedding_cfg()

load_dotenv(Path(__file__).resolve().parent.parent.parent / _cfg["paths"]["env_file"])
openai.api_key   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

OPENAI_MODEL = _emb_cfg["openai_model"]
HF_MODEL     = _emb_cfg["hf_model"]
BATCH_SIZE   = _emb_cfg["batch_size"]
MAX_CHARS    = _emb_cfg["max_chars"]
SLEEP_SEC    = _emb_cfg["sleep_sec"]

# HuggingFace 모델은 필요할 때만 로드 (메모리 절약)
_hf_model = None

def _get_hf_model():
    global _hf_model
    if _hf_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  HuggingFace 모델 로드 중: {HF_MODEL}")
        _hf_model = SentenceTransformer(HF_MODEL)
    return _hf_model


# ══════════════════════════════════════════════
# 1. OpenAI 임베딩 (변경 없음)
# ══════════════════════════════════════════════

def _truncate(text: str) -> str:
    if not isinstance(text, str):
        return " "
    return text[:MAX_CHARS]


def _openai_batch(batch: list[str]) -> list[list[float]]:
    """단일 배치 OpenAI 임베딩 — Rate limit / BadRequest 재시도"""
    while True:
        try:
            response = openai.embeddings.create(input=batch, model=OPENAI_MODEL)
            time.sleep(SLEEP_SEC)
            return [item.embedding for item in response.data]
        except openai.RateLimitError:
            tqdm.write("Rate limit → 5초 대기...")
            time.sleep(5)
        except openai.BadRequestError as e:
            tqdm.write(f"BadRequest → 개별 처리: {e}")
            results = []
            for t in batch:
                try:
                    r = openai.embeddings.create(input=[t[:10000]], model=OPENAI_MODEL)
                    results.append(r.data[0].embedding)
                except Exception as e2:
                    tqdm.write(f"개별 실패 → 빈 벡터: {e2}")
                    results.append([0.0] * 1536)
            return results


def _embed_openai(texts: list[str], use_ckpt: bool, ckpt_dir: Path) -> np.ndarray:
    """OpenAI 전체 임베딩 — 체크포인트 유무에 따라 분기"""
    texts = [_truncate(t) for t in texts]

    # 체크포인트 없는 버전 (소용량)
    if not use_ckpt:
        all_emb = []
        total   = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        start   = time.time()
        for i in tqdm(range(total), desc="  OpenAI 임베딩"):
            batch = texts[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            all_emb.extend(_openai_batch(batch))
        print(f"  소요: {(time.time()-start)/60:.1f}분")
        return np.array(all_emb, dtype="float32")

    # 체크포인트 있는 버전 (대용량)
    ckpt_dir.mkdir(exist_ok=True)
    saved       = sorted(ckpt_dir.glob("chunk_*.npy"))
    start_idx   = len(saved) * 1000
    chunk_emb   = []
    chunk_idx   = len(saved)
    total_batch = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    start       = time.time()

    print(f"  체크포인트 {len(saved)}개 확인 → {start_idx:,}번부터 재개")

    for i in tqdm(range(start_idx // BATCH_SIZE, total_batch), desc="  OpenAI 임베딩"):
        batch = texts[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        chunk_emb.extend(_openai_batch(batch))

        if len(chunk_emb) >= 1000:
            f = ckpt_dir / f"chunk_{chunk_idx:04d}.npy"
            np.save(str(f), np.array(chunk_emb[:1000], dtype="float32"))
            tqdm.write(f"  청크 저장: {f.name}")
            chunk_emb = chunk_emb[1000:]
            chunk_idx += 1

    if chunk_emb:
        f = ckpt_dir / f"chunk_{chunk_idx:04d}.npy"
        np.save(str(f), np.array(chunk_emb, dtype="float32"))

    all_chunks  = sorted(ckpt_dir.glob("chunk_*.npy"))
    total_rows  = sum(np.load(str(c), mmap_mode='r').shape[0] for c in all_chunks)
    n_dim       = np.load(str(all_chunks[0]), mmap_mode='r').shape[1]

    merged_path = ckpt_dir / "merged.npy"
    embeddings  = np.lib.format.open_memmap(
        str(merged_path), mode='w+', dtype='float32', shape=(total_rows, n_dim)
    )
    idx = 0
    for c in tqdm(all_chunks, desc="  청크 병합"):
        chunk = np.load(str(c), mmap_mode='r')
        embeddings[idx: idx + chunk.shape[0]] = chunk
        idx += chunk.shape[0]
        del chunk

    print(f"  소요: {(time.time()-start)/60:.1f}분")
    return embeddings


# ══════════════════════════════════════════════
# 2. HuggingFace 임베딩 (변경 없음)
# ══════════════════════════════════════════════

def _embed_hf(texts: list[str]) -> np.ndarray:
    """HuggingFace SentenceTransformer 임베딩"""
    model = _get_hf_model()
    print(f"  HuggingFace 임베딩 시작... ({len(texts):,}개)")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,  # ← 정규화 True → dotproduct = cosine 유사도
        batch_size=64,
    )
    return embeddings.astype(np.float32)


# ══════════════════════════════════════════════
# 3. Pinecone 저장 (FAISS → Pinecone으로 교체)
# ══════════════════════════════════════════════

def _get_pinecone_client() -> Pinecone:
    """Pinecone 클라이언트 반환"""
    if not PINECONE_API_KEY:
        raise EnvironmentError(
            "PINECONE_API_KEY가 없습니다.\n"
            ".env 파일에 PINECONE_API_KEY=your_key 를 추가해주세요."
        )
    return Pinecone(api_key=PINECONE_API_KEY)


def _get_or_create_index(pc: Pinecone, index_name: str, dim: int, metric: str):
    """
    Pinecone 인덱스 없으면 생성, 있으면 metric 검증 후 반환

    Args:
        pc         : Pinecone 클라이언트
        index_name : 인덱스명 (예: "nolit-boardgame")
        dim        : 벡터 차원수 (OpenAI=1536, HuggingFace=768)
        metric     : 유사도 방식 (OpenAI="cosine", HuggingFace="dotproduct")

    Note:
        Pinecone은 인덱스 생성 후 metric 변경 불가.
        이미 존재하는 인덱스의 metric이 다르면 경고 출력.
    """
    existing_names = pc.list_indexes().names()

    if index_name not in existing_names:
        print(f"  🆕 인덱스 생성 중: {index_name} (dim={dim}, metric={metric})")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"  ✅ 인덱스 생성 완료: {index_name}")
    else:
        # 이미 존재하면 metric 검증
        existing_info   = pc.describe_index(index_name)
        existing_metric = existing_info.metric

        if existing_metric != metric:
            print(
                f"  ⚠️  경고: [{index_name}] 기존 metric={existing_metric}, "
                f"요청 metric={metric} — 기존 인덱스 그대로 사용합니다.\n"
                f"     metric을 바꾸려면 Pinecone 콘솔에서 인덱스를 삭제 후 재생성하세요."
            )
        else:
            print(f"  ✔  인덱스 이미 존재: {index_name} (metric={existing_metric})")

    return pc.Index(index_name)


def _clean_metadata(meta: dict) -> dict:
    """
    Pinecone 메타데이터 정제

    Pinecone이 저장 못 하는 타입 변환:
      None       → 빈 문자열 ""
      NaN        → 빈 문자열 ""
      list       → " | " 로 합친 문자열
      dict       → 문자열로 변환 (예: category_rank)
      그 외      → 그대로
    """
    cleaned = {}
    for key, val in meta.items():
        if val is None:
            cleaned[key] = ""
        elif isinstance(val, float) and math.isnan(val):
            cleaned[key] = ""
        elif isinstance(val, list):
            cleaned[key] = " | ".join(str(v) for v in val)
        elif isinstance(val, dict):
            cleaned[key] = str(val)   # {"Overall": 1.0, ...} → 문자열로 저장
        else:
            cleaned[key] = val
    return cleaned


def _save_pinecone(
    embeddings  : np.ndarray,
    metas       : list[dict],
    index_name  : str,
    namespace   : str,
    source_name : str,
    metric      : str,
    batch_size  : int = 100,
):
    """
    벡터 + 메타데이터를 Pinecone에 저장

    Args:
        embeddings  : shape (N, dim) float32 벡터
        metas       : 메타데이터 딕셔너리 리스트
        index_name  : Pinecone 인덱스명 (예: "nolit-boardgame")
        namespace   : 네임스페이스 (예: "bgg_stats")
        source_name : 벡터 ID 접두사용 (예: "bgg_stats_1")
        metric      : 유사도 방식 (예: "cosine" | "dotproduct")
        batch_size  : 한 번에 upsert할 벡터 수 (기본 100)
    """
    pc    = _get_pinecone_client()
    dim   = embeddings.shape[1]
    total = len(metas)

    # 인덱스 없으면 생성, 있으면 metric 검증
    index = _get_or_create_index(pc, index_name, dim, metric)

    print(f"  Pinecone 저장 시작: {total:,}개 → {index_name}/{namespace}")

    vectors = []
    for i in tqdm(range(total), desc="  Pinecone upsert"):
        vectors.append({
            "id"      : f"{source_name}_{i + 1}",      # 예: bgg_stats_1
            "values"  : embeddings[i].tolist(),          # float32 → list[float]
            "metadata": _clean_metadata(metas[i]),       # 정제된 메타데이터
        })

        # batch_size마다 한 번씩 upsert
        if len(vectors) == batch_size:
            index.upsert(vectors=vectors, namespace=namespace)
            vectors = []

    # 남은 벡터 마저 upsert
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)

    print(f"  ✅ Pinecone 저장 완료: {total:,}개 ({index_name}/{namespace})")


# ══════════════════════════════════════════════
# 4. 실행 함수 (외부 호출용)
# ══════════════════════════════════════════════

def run_embedding(source_name: str):
    """
    단일 데이터소스 임베딩 실행.
    management command / 외부 스크립트에서 호출.
    """
    cfg     = get_config()
    sources = cfg["sources"]

    if source_name not in sources:
        raise ValueError(
            f"알 수 없는 소스: '{source_name}'\n"
            f"사용 가능: {list(sources.keys())}"
        )

    src_cfg  = sources[source_name]
    engine   = src_cfg.get("engine", "openai")
    use_ckpt = src_cfg.get("use_ckpt", False)
    data_dir = get_data_dir()

    # config.py에서 Pinecone 인덱스명 + 네임스페이스 가져오기
    index_name, namespace = get_source_pinecone(source_name)

    print(f"\n{'='*52}")
    print(f"  [{source_name}]  engine={engine}  ckpt={use_ckpt}")
    print(f"  → Pinecone: {index_name} / {namespace}")
    print(f"{'='*52}")

    # 1. 데이터 로드
    texts, metas = load_source(source_name)
    print(f"  로드 완료: {len(texts):,}개")

    # 2. 임베딩 생성
    #    engine에 따라 임베딩 방식과 metric이 달라짐
    #    - hf      : normalize=True → dotproduct (= cosine 유사도)
    #    - openai  : 정규화 없음    → cosine
    ckpt_dir = data_dir / f"ckpt_{source_name}"

    if engine == "hf":
        embeddings = _embed_hf(texts)
        metric     = "dotproduct"   # 정규화된 벡터 → 내적
    else:
        embeddings = _embed_openai(texts, use_ckpt, ckpt_dir)
        metric     = "cosine"       # OpenAI → 코사인

    print(f"  임베딩 shape: {embeddings.shape}  metric: {metric}")

    # 3. Pinecone 저장
    _save_pinecone(
        embeddings  = embeddings,
        metas       = metas,
        index_name  = index_name,
        namespace   = namespace,
        source_name = source_name,
        metric      = metric,
    )

    print(f"\n  ✅ [{source_name}] 완료!\n")


def run_all():
    """config.yaml에 정의된 모든 데이터소스 순차 임베딩"""
    sources = list_sources()
    print(f"\n전체 임베딩 시작: {sources}\n")
    for source_name in sources:
        run_embedding(source_name)


# ══════════════════════════════════════════════
# 5. retriever.py에서 사용하는 로드 함수
# ══════════════════════════════════════════════

def load_index(source_name: str) -> tuple:
    """
    Pinecone 인덱스 연결 반환.
    retriever.py에서 검색 시 사용.

    Returns:
        (pinecone.Index, namespace: str)

    변경 전: (faiss.Index, list[dict])
    변경 후: (pinecone.Index, namespace)
        → retriever.py에서 metas를 직접 들고 다니지 않아도 됨
          Pinecone이 메타데이터를 검색 결과에 포함해서 반환하기 때문
    """
    index_name, namespace = get_source_pinecone(source_name)

    pc    = _get_pinecone_client()
    index = pc.Index(index_name)

    return index, namespace


def get_query_embedding(text: str, engine: str = "openai") -> np.ndarray:
    """
    쿼리 텍스트 임베딩 생성.
    retriever.py에서 검색 쿼리 임베딩 시 사용.

    Args:
        text   : 검색 쿼리 텍스트
        engine : "openai" | "hf"

    Returns:
        shape (1, dim) float32 ndarray
    """
    if engine == "hf":
        model = _get_hf_model()
        vec   = model.encode([text], normalize_embeddings=True)
        return vec.astype(np.float32)
    else:
        response = openai.embeddings.create(input=[text], model=OPENAI_MODEL)
        vec      = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)   # 검색 전 정규화 (코사인 유사도 정확도 향상)
        return vec