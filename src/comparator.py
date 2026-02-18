from dataclasses import dataclass
from typing import List, Dict

from sentence_transformers import SentenceTransformer

from src.skills_reference import SkillReference
from src.scoring import compute_scores
from src.semantic_matching import semantic_compare, SemanticGap


@dataclass
class CompareResult:
    scores: Dict[str, float]
    missing_skills: List[str]
    matched_skills: List[str]
    semantic_gaps: List[SemanticGap]


def compare_texts(
    cv_text: str,
    job_text: str,
    skill_ref: SkillReference,
    emb_model: SentenceTransformer,
    semantic_threshold: float = 0.35,
    chunk_size_words: int = 220,
    overlap_words: int = 60,
) -> CompareResult:
    # 1) Skills-based (explicable)
    cv_skills = skill_ref.extract(cv_text)
    job_skills = skill_ref.extract(job_text)

    matched = sorted(list(cv_skills & job_skills))
    missing = sorted(list(job_skills - cv_skills))

    # 2) Sémantique (embeddings + FAISS)
    sem = semantic_compare(
        cv_text=cv_text,
        job_text=job_text,
        model=emb_model,
        threshold=semantic_threshold,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
        top_k=1,
    )

    # 3) Score global (priorité sémantique)
    scores = compute_scores(
        cv_text=cv_text,
        job_text=job_text,
        cv_skills=cv_skills,
        job_skills=job_skills,
        semantic_score=sem.semantic_score,
        semantic_coverage=sem.semantic_coverage,
    )

    return CompareResult(
        scores=scores,
        missing_skills=missing,
        matched_skills=matched,
        semantic_gaps=sem.gaps,
    )
