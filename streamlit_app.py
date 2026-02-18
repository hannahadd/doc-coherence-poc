import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.pdf_utils import extract_text_from_pdf_bytes
from src.skills_reference import SkillReference
from src.comparator import compare_texts


# Force offline pour HF/Transformers (si le modèle est local, aucun souci)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

st.set_page_config(page_title="PDF coherence checker (Offline)", layout="wide")
st.title("PDF Coherence Checker (Offline)")
st.caption("Compare CV vs offre d'emploi : score sémantique, manquants, conseils, export JSON.")


# Supporte data/ OU assets/data selon ton arborescence
ROOT = Path(__file__).parent
DATA_DIR = (ROOT / "assets" / "data") if (ROOT / "assets" / "data").exists() else (ROOT / "data")
MODELS_DIR = (ROOT / "assets" / "models") if (ROOT / "assets" / "models").exists() else (ROOT / "models")


def _clip(text: str, max_chars: int = 6000) -> str:
    return (text or "")[:max_chars]


def run_ollama(model: str, prompt: str, timeout_s: int = 90) -> tuple[Optional[str], Optional[str]]:
    try:
        p = subprocess.run(
            ["ollama", "run", model, prompt],
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        if p.returncode != 0:
            return None, (p.stderr or "").strip() or f"Ollama exited with {p.returncode}"
        out = (p.stdout or "").strip()
        return out if out else None, None
    except FileNotFoundError:
        return None, "Ollama n'est pas installé (commande 'ollama' introuvable)."
    except subprocess.TimeoutExpired:
        return None, "Timeout Ollama (prompt trop long ou modèle lent)."


def fallback_advice(missing_skills: list[str], gaps_text: str, score_pct: float) -> str:
    base = [f"Score: {score_pct}%."]
    if gaps_text:
        base.append("Points probablement non couverts (sémantique):")
        base.append(gaps_text)
    if missing_skills:
        base.append("")
        base.append("Compétences manquantes détectées (lexique): " + ", ".join(missing_skills[:12]))
    base.append("")
    base.append("Conseils:")
    base.append("- Mettre en avant les éléments clés demandés dans l'offre (section Compétences en haut).")
    base.append("- Ajouter des preuves : projets/missions, outils, résultats mesurables.")
    base.append("- Ne pas inventer : préciser 'projet académique' / 'notions' si nécessaire.")
    return "\n".join(base)


def generate_cv_advice_llm(
    cv_text: str,
    job_text: str,
    missing_skills: list[str],
    semantic_gaps_snippets: list[str],
    score_pct: float,
    model: str,
) -> str:
    missing_str = ", ".join(missing_skills[:25]) if missing_skills else "(aucune)"
    gaps_str = "\n- " + "\n- ".join(semantic_gaps_snippets[:8]) if semantic_gaps_snippets else "(aucun)"

    prompt = f"""
Tu es un expert recrutement. Compare un CV à une offre d'emploi.
Objectif: expliquer le manque de cohérence et donner des conseils concrets.

Contraintes:
- Ne pas inventer d'expérience.
- Conseils actionnables (quoi changer, où le mettre, exemples de formulation).
- Réponse courte et structurée.

Score actuel: {score_pct}%
Compétences manquantes (lexique): {missing_str}

Gaps sémantiques (extraits offre peu couverts):
{gaps_str}

OFFRE (extrait):
{_clip(job_text, 2600)}

CV (extrait):
{_clip(cv_text, 2600)}

Rédige:
1) Diagnostic (3-5 points)
2) Conseils (6-10 bullets)
3) 2 exemples de formulations courtes (CV) pour 1-2 manquants/gaps
""".strip()

    out, err = run_ollama(model=model, prompt=prompt, timeout_s=90)
    if out:
        return out

    return fallback_advice(
        missing_skills=missing_skills,
        gaps_text="\n".join([f"- {s}" for s in semantic_gaps_snippets[:8]]),
        score_pct=score_pct,
    ) + (f"\n\n(LLM indisponible: {err})" if err else "")


@st.cache_resource
def load_skill_ref(p: str):
    return SkillReference.load(p)


@st.cache_resource
def load_embedding_model(model_path: str) -> SentenceTransformer:
    # Important: donne un chemin local (MODELS_DIR/...) pour rester offline
    return SentenceTransformer(model_path)


def parse_label(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip().lower()
    if s in {"1", "fit", "good fit", "yes"}:
        return 1
    if s in {"0", "no fit", "not fit", "bad fit", "no"}:
        return 0
    m = re.search(r"\b([01])\b", s)
    return int(m.group(1)) if m else None


def render_result(result, threshold, cv_text: str, job_text: str, use_llm: bool, ollama_model: str, show_label=None):
    score_pct = result.scores["score_pct"]
    coherent = score_pct >= threshold

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{score_pct}%")
    c2.metric("Semantic coverage", f"{round(result.scores['semantic_coverage'] * 100, 1)}%")
    c3.metric("Semantic score", f"{round(result.scores['semantic_score'] * 100, 1)}%")
    c4.metric("Verdict", "Cohérent" if coherent else "À risque")

    if show_label is not None:
        st.caption(f"Label dataset (0/1): {show_label}")

    # Signaux secondaires (discrets)
    st.caption(
        f"Skill coverage: {round(result.scores['skill_coverage'] * 100, 1)}% | "
        f"TF-IDF similarity: {round(result.scores['text_similarity'] * 100, 1)}%"
    )

    st.divider()

    # 1) Gaps sémantiques (le plus important pour doc coherence)
    st.subheader(f"Gaps sémantiques (extraits offre peu couverts) ({len(result.semantic_gaps)})")
    if result.semantic_gaps:
        rows = []
        for g in result.semantic_gaps[:20]:
            rows.append(
                {
                    "best_sim": round(g.best_sim, 3),
                    "offre_extrait": g.job_snippet,
                    "cv_plus_proche": g.best_cv_snippet,
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        semantic_gaps_snippets = [g.job_snippet for g in result.semantic_gaps]
    else:
        st.write(["(aucun gap détecté au seuil actuel)"])
        semantic_gaps_snippets = []

    st.divider()

    # 2) Skills manquantes (utile mais dépend du lexique)
    st.subheader(f"Compétences manquantes (lexique) ({len(result.missing_skills)})")
    st.write(result.missing_skills if result.missing_skills else ["(aucune)"])
    st.caption(f"Compétences matchées: {len(result.matched_skills)}")

    st.divider()

    # 3) Conseils
    st.subheader("Conseils pour améliorer la cohérence du CV")
    if use_llm:
        with st.spinner("Génération des conseils (LLM local)..."):
            advice = generate_cv_advice_llm(
                cv_text=cv_text,
                job_text=job_text,
                missing_skills=result.missing_skills,
                semantic_gaps_snippets=semantic_gaps_snippets,
                score_pct=score_pct,
                model=ollama_model,
            )
    else:
        advice = fallback_advice(result.missing_skills, "\n".join([f"- {s}" for s in semantic_gaps_snippets[:8]]), score_pct)

    st.write(advice)

    report = {
        "scores": result.scores,
        "threshold_pct": threshold,
        "verdict": "coherent" if coherent else "at_risk",
        "missing_skills": result.missing_skills,
        "matched_skills": result.matched_skills,
        "semantic_gaps": [
            {"best_sim": g.best_sim, "job_snippet": g.job_snippet, "best_cv_snippet": g.best_cv_snippet}
            for g in result.semantic_gaps
        ],
        "advice": advice,
    }

    st.download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="coherence_report.json",
        mime="application/json",
        use_container_width=True,
    )


with st.sidebar:
    st.header("Paramètres")
    skill_path = st.text_input("skills_reference.json", value=str(DATA_DIR / "skills_reference.json"))

    st.markdown("---")
    st.subheader("Sémantique (embeddings)")
    # IMPORTANT: mets un chemin local vers un modèle déjà présent sur le disque
    default_model = str(MODELS_DIR / "all-MiniLM-L6-v2") if MODELS_DIR.exists() else "models/all-MiniLM-L6-v2"
    emb_model_path = st.text_input("Embedding model (local path)", value=default_model)
    semantic_threshold = st.slider("Seuil sémantique", 0.0, 1.0, 0.35, 0.01)
    chunk_size_words = st.slider("Chunk size (words)", 80, 400, 220, 10)
    overlap_words = st.slider("Overlap (words)", 0, 200, 60, 10)

    st.markdown("---")
    verdict_threshold = st.slider("Seuil 'cohérent' (%)", 0, 100, 60, 5)

    st.markdown("---")
    st.subheader("Conseils (local)")
    use_llm = st.checkbox("Générer des conseils avec un LLM local (Ollama)", value=False)
    ollama_model = st.text_input("Modèle Ollama", value="llama3.2:3b")

    st.markdown("---")
    st.caption("Offline only: aucune requête réseau, aucun cloud.")


try:
    skill_ref = load_skill_ref(skill_path)
except Exception as e:
    st.error(f"Impossible de charger le référentiel de compétences: {e}")
    st.stop()

# Chargement modèle embeddings (chemin local)
try:
    emb_model = load_embedding_model(emb_model_path)
except Exception as e:
    st.error(
        "Impossible de charger le modèle d'embedding.\n"
        f"- Chemin: {emb_model_path}\n"
        f"- Erreur: {e}\n\n"
        "Astuce: télécharge le modèle sur une machine autorisée puis copie le dossier dans ./models/ ou ./assets/models/."
    )
    st.stop()


tab1, tab2 = st.tabs(["Compare PDFs", "Demo dataset"])

# ----------------------
# TAB 1: PDFs
# ----------------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        cv_pdf = st.file_uploader("Upload CV (PDF)", type=["pdf"], key="cv_pdf")
    with col2:
        job_pdf = st.file_uploader("Upload offre (PDF)", type=["pdf"], key="job_pdf")

    run = st.button("Comparer", type="primary", use_container_width=True, key="run_pdf")

    if run:
        if not cv_pdf or not job_pdf:
            st.error("Il faut uploader les 2 PDFs.")
            st.stop()

        cv_ex = extract_text_from_pdf_bytes(cv_pdf.getvalue())
        job_ex = extract_text_from_pdf_bytes(job_pdf.getvalue())

        if not cv_ex.text.strip() or not job_ex.text.strip():
            st.error("Texte vide: PDF scanné ou extraction impossible (OCR non inclus).")
            st.stop()

        result = compare_texts(
            cv_text=cv_ex.text,
            job_text=job_ex.text,
            skill_ref=skill_ref,
            emb_model=emb_model,
            semantic_threshold=semantic_threshold,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )
        render_result(
            result,
            verdict_threshold,
            cv_text=cv_ex.text,
            job_text=job_ex.text,
            use_llm=use_llm,
            ollama_model=ollama_model,
            show_label=None,
        )

# ----------------------
# TAB 2: Dataset demo
# ----------------------
with tab2:
    st.info("Démo sans PDF: couples (resume, job_description) depuis le fichier XLSX dans data/.")

    @st.cache_data
    def load_demo_df():
        p = DATA_DIR / "huggingface_resume_job_fit_RAW.xlsx"
        df = pd.read_excel(p)
        df.columns = [c.strip().lower() for c in df.columns]
        df["_label"] = df["label"].apply(parse_label) if "label" in df.columns else None
        return df

    df = load_demo_df()

    colA, colB, colC = st.columns([2, 2, 1])
    with colA:
        label_filter = st.selectbox("Filtre label", options=["all", "0", "1"], index=0)
    with colB:
        idx = st.number_input("Index exemple", min_value=0, max_value=len(df) - 1, value=0, step=1)
    with colC:
        random_pick = st.button("Random")

    if label_filter != "all":
        sub = df[df["_label"] == int(label_filter)]
        if len(sub) == 0:
            st.warning("Aucun exemple pour ce label.")
            st.stop()
        row = sub.sample(1).iloc[0] if random_pick else sub.iloc[int(idx) % len(sub)]
    else:
        row = df.sample(1).iloc[0] if random_pick else df.iloc[int(idx)]

    label = row["_label"] if "_label" in row else None

    cv_text = str(row.get("resume_text", ""))
    job_text = str(row.get("job_description_text", ""))

    run2 = st.button("Comparer (dataset)", type="primary", use_container_width=True, key="run_dataset")

    if run2:
        result = compare_texts(
            cv_text=cv_text,
            job_text=job_text,
            skill_ref=skill_ref,
            emb_model=emb_model,
            semantic_threshold=semantic_threshold,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )
        render_result(
            result,
            verdict_threshold,
            cv_text=cv_text,
            job_text=job_text,
            use_llm=use_llm,
            ollama_model=ollama_model,
            show_label=label,
        )
