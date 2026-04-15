import re
from dataclasses import dataclass
from typing import List
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
    semantic_score: float      # moyenne des best_sim (0..1)
    semantic_coverage: float   # fraction de chunks job "couverts" (best_sim >= threshold)
    gaps: List[SemanticGap]


def _split_sentences(text: str) -> List[str]:
    """Découpe en phrases via ponctuation. Pas de dépendance NLTK."""
    text = (text or "").strip()
    if not text:
        return []
    # Coupe sur . ! ? suivi d'un espace + majuscule ou fin de chaîne
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_semantic(
    text: str,
    model: SentenceTransformer,
    breakpoint_percentile: int = 95,
    min_sentences: int = 2,
    max_sentences: int = 20,
) -> List[str]:
    """
    Chunking sémantique : coupe le texte aux ruptures thématiques.

    Principe :
    - On encode chaque phrase individuellement.
    - On calcule la similarité cosinus entre phrases consécutives.
    - Les points où la similarité chute (sous le percentile bas) sont
      des ruptures de sujet → on coupe là.

    breakpoint_percentile : plus la valeur est haute, plus on coupe souvent
                            (chunks plus petits et plus homogènes).
    """
    sentences = _split_sentences(text)

    # Texte trop court : un seul chunk
    if len(sentences) <= min_sentences:
        return [text] if text.strip() else []

    # Encode toutes les phrases (batch, normalisé)
    vecs = model.encode(
        sentences,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    vecs = np.asarray(vecs, dtype=np.float32)

    # Similarité cosinus entre chaque paire de phrases consécutives
    # (vecteurs normalisés => dot product = cosine)
    sims = np.array([
        float(np.dot(vecs[i], vecs[i + 1]))
        for i in range(len(vecs) - 1)
    ])

    # Seuil : on coupe où la similarité est la plus basse
    # (les (100 - breakpoint_percentile) % de transitions les moins similaires)
    cut_threshold = np.percentile(sims, 100 - breakpoint_percentile)

    # Construction des chunks
    chunks: List[str] = []
    current: List[str] = [sentences[0]]

    for i, sentence in enumerate(sentences[1:], start=1):
        prev_sim = sims[i - 1]
        is_breakpoint = prev_sim < cut_threshold
        exceeds_max = len(current) >= max_sentences

        if (is_breakpoint and len(current) >= min_sentences) or exceeds_max:
            chunks.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Encodage embeddings normalisés => cosine similarity = dot product."""
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def build_faiss_index(vectors: np.ndarray):
    """Index FlatIP (inner product) sur vecteurs normalisés => cosine similarity."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def semantic_compare(
    cv_text: str,
    job_text: str,
    model: SentenceTransformer,
    threshold: float = 0.35,
    breakpoint_percentile: int = 95,
    top_k: int = 1,
) -> SemanticCompareResult:
    """
    Compare job vs CV sémantiquement :
    - Chunking sémantique (ruptures thématiques) sur les deux textes
    - Indexation des chunks CV via FAISS
    - Pour chaque chunk offre, on trouve le chunk CV le plus proche
    - Si best_sim < threshold => gap sémantique
    """
    job_chunks = chunk_text_semantic(job_text, model, breakpoint_percentile)
    cv_chunks = chunk_text_semantic(cv_text, model, breakpoint_percentile)

    if not job_chunks or not cv_chunks:
        return SemanticCompareResult(semantic_score=0.0, semantic_coverage=0.0, gaps=[])

    job_vec = embed_texts(model, job_chunks)
    cv_vec = embed_texts(model, cv_chunks)

    if HAS_FAISS:
        index = build_faiss_index(cv_vec)
        sims, ids = index.search(job_vec, top_k)
        best_sims = sims[:, 0]
        best_ids = ids[:, 0]
    else:
        # fallback bruteforce (OK si nb chunks raisonnable)
        sims_full = job_vec @ cv_vec.T
        best_ids = np.argmax(sims_full, axis=1)
        best_sims = sims_full[np.arange(sims_full.shape[0]), best_ids]

    semantic_score = float(np.mean(best_sims))
    semantic_coverage = float(np.mean(best_sims >= threshold))

    gaps: List[SemanticGap] = []
    for idx in np.argsort(best_sims):  # du moins couvert au plus couvert
        sim = float(best_sims[idx])
        if sim >= threshold:
            continue
        job_snip = job_chunks[idx][:320].strip()
        cv_snip = cv_chunks[int(best_ids[idx])][:320].strip() if cv_chunks else ""
        gaps.append(SemanticGap(best_sim=sim, job_snippet=job_snip, best_cv_snippet=cv_snip))

    return SemanticCompareResult(
        semantic_score=semantic_score,
        semantic_coverage=semantic_coverage,
        gaps=gaps,
    )
