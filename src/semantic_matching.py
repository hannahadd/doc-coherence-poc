from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

from sentence_transformers import SentenceTransformer


@dataclass
class SemanticGap:
    best_sim: float
    job_snippet: str
    best_cv_snippet: str


@dataclass
class SemanticCompareResult:
    semantic_score: float          # moyenne des best_sim (0..1)
    semantic_coverage: float       # fraction de chunks job "couverts" (best_sim >= threshold)
    gaps: List[SemanticGap]


def chunk_text(text: str, chunk_size_words: int = 220, overlap_words: int = 60) -> List[str]:
    """
    Chunk simple par mots (POC robuste).
    - chunk_size_words: taille approx du chunk
    - overlap_words: recouvrement pour ne pas couper une idée en 2
    """
    words = (text or "").split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Encodage embeddings normalisés => cosine similarity = dot product.
    """
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def build_faiss_index(vectors: np.ndarray):
    """
    Index FlatIP (inner product) sur vecteurs normalisés => cosine similarity.
    """
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # exact, simple, stable pour POC
    index.add(vectors)
    return index


def semantic_compare(
    cv_text: str,
    job_text: str,
    model: SentenceTransformer,
    threshold: float = 0.35,
    chunk_size_words: int = 220,
    overlap_words: int = 60,
    top_k: int = 1,
) -> SemanticCompareResult:
    """
    Compare job vs CV sémantiquement :
    - On découpe l'offre en chunks (ce qu'on veut "couvrir")
    - On indexe les chunks du CV
    - Pour chaque chunk offre, on trouve le chunk CV le plus proche
    - Si best_sim < threshold => gap sémantique
    """
    job_chunks = chunk_text(job_text, chunk_size_words, overlap_words)
    cv_chunks = chunk_text(cv_text, chunk_size_words, overlap_words)

    if not job_chunks or not cv_chunks:
        return SemanticCompareResult(semantic_score=0.0, semantic_coverage=0.0, gaps=[])

    job_vec = embed_texts(model, job_chunks)
    cv_vec = embed_texts(model, cv_chunks)

    if HAS_FAISS:
        index = build_faiss_index(cv_vec)
        sims, ids = index.search(job_vec, top_k)  # sims: (n_job,k), ids:(n_job,k)
        best_sims = sims[:, 0]
        best_ids = ids[:, 0]
    else:
        # fallback bruteforce (OK si nb chunks raisonnable)
        sims_full = job_vec @ cv_vec.T  # dot product sur vects normalisés => cosine
        best_ids = np.argmax(sims_full, axis=1)
        best_sims = sims_full[np.arange(sims_full.shape[0]), best_ids]

    semantic_score = float(np.mean(best_sims))
    semantic_coverage = float(np.mean(best_sims >= threshold))

    gaps: List[SemanticGap] = []
    # on renvoie les chunks offre les moins couverts
    order = np.argsort(best_sims)  # ascending
    for idx in order:
        sim = float(best_sims[idx])
        if sim >= threshold:
            continue
        job_snip = job_chunks[idx][:320].strip()
        cv_snip = cv_chunks[int(best_ids[idx])][:320].strip() if len(cv_chunks) > 0 else ""
        gaps.append(SemanticGap(best_sim=sim, job_snippet=job_snip, best_cv_snippet=cv_snip))

    return SemanticCompareResult(
        semantic_score=semantic_score,
        semantic_coverage=semantic_coverage,
        gaps=gaps,
    )
