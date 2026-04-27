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

    # TF-IDF (signal complémentaire)
    tfidf_sim = _safe_cosine(job_text, cv_text)

    # ── Scores affichés à l'utilisateur RH ────────────────────────────────────
    # skill_score : proportion des compétences explicites de l'offre trouvées dans le CV
    skill_score = skill_cov

    # semantic_display_score : alignement du vocabulaire et des expériences
    # (moyenne embeddings + TF-IDF pour lisser les artefacts de chaque méthode)
    semantic_display_score = (semantic_score + tfidf_sim) / 2

    # ── Score global (usage interne : classement uniquement) ──────────────────
    score = skill_score * 0.40 + semantic_display_score * 0.60

    return {
        "semantic_coverage": float(semantic_coverage),
        "semantic_score": float(semantic_score),
        "skill_coverage": float(skill_cov),
        "text_similarity": float(tfidf_sim),
        "skill_score": float(skill_score),
        "semantic_display_score": float(semantic_display_score),
        "score": float(score),
        "score_pct": round(score * 100, 1),
    }
