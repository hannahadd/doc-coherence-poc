from typing import Dict, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _safe_cosine(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0

    vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X = vect.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0][0])


def compute_scores(
    cv_text: str,
    job_text: str,
    cv_skills: Set[str],
    job_skills: Set[str],
    semantic_score: float,
    semantic_coverage: float,
) -> Dict[str, float]:
    # skills coverage (utile mais secondaire)
    if job_skills:
        skill_cov = len(cv_skills & job_skills) / len(job_skills)
    else:
        skill_cov = 0.0

    # TF-IDF (debug / signal complémentaire)
    tfidf_sim = _safe_cosine(job_text, cv_text)

    # Score final : sémantique prioritaire + un peu de skills
    # - semantic_coverage: répond à "est-ce que les points de l'offre sont couverts"
    # - semantic_score: match global moyen
    # - skill_cov: garde un signal explicable sur les termes attendus
    score = 0.55 * semantic_coverage + 0.30 * semantic_score + 0.15 * skill_cov

    return {
        "semantic_coverage": float(semantic_coverage),
        "semantic_score": float(semantic_score),
        "skill_coverage": float(skill_cov),
        "text_similarity": float(tfidf_sim),
        "score": float(score),
        "score_pct": round(score * 100, 1),
    }
