import os
from dotenv import load_dotenv

# Charge .env avant tout le reste (MISTRAL_API_KEY, etc.)
# override=False : les variables d'environnement existantes ont priorité.
load_dotenv(override=False)

# Doit être défini AVANT l'import de torch/faiss pour éviter le conflit libomp
# (nécessaire sur macOS ; sans effet négatif sur Linux/Docker)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.doc_reader import extract_from_bytes
from src.skills_reference import SkillReference
from src.pipeline import analyze_candidate
from src.job_offer_parser import extract_profil_section

# ── Charte TriCV ──────────────────────────────────────────────────────────────
from styles import CSS, HEADER_HTML, TOPBAR_HTML, SIDEBAR_BRAND_HTML


# Force offline pour HF/Transformers (si le modèle est local, aucun souci)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

st.set_page_config(
    page_title="TriCV — Analyse CV / Offre",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Injection CSS ─────────────────────────────────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)

# ── Topbar persistant (reste visible même quand la sidebar est fermée) ────────
st.markdown(TOPBAR_HTML, unsafe_allow_html=True)

# ── En-tête TriCV ─────────────────────────────────────────────────────────────
st.markdown(HEADER_HTML, unsafe_allow_html=True)


# Supporte data/ OU assets/data selon ton arborescence
ROOT = Path(__file__).parent
DATA_DIR = (ROOT / "assets" / "data") if (ROOT / "assets" / "data").exists() else (ROOT / "data")
MODELS_DIR = (ROOT / "assets" / "models") if (ROOT / "assets" / "models").exists() else (ROOT / "models")



@st.cache_resource
def load_skill_ref(base_path: str, company_path: str):
    paths = [p for p in [base_path, company_path] if p.strip()]
    return SkillReference.load_multi(paths)


@st.cache_resource
def load_embedding_model(model_path: str) -> SentenceTransformer:
    # Important: donne un chemin local (MODELS_DIR/...) pour rester offline
    return SentenceTransformer(model_path)




def _render_document_preview(name: str, data: bytes) -> None:
    """Affiche le document : PDF (pages rendues en image), image, ou texte extrait."""
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if ext == "pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        for i, page in enumerate(doc):
            # Zoom 2.0 → 144 dpi : lisible et net à toutes les largeurs de conteneur
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            st.image(pix.tobytes("png"), use_container_width=True,
                     caption=f"Page {i + 1} / {len(doc)}")
    elif ext in {"jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp"}:
        st.image(data, use_container_width=True)
    else:
        ex = extract_from_bytes(data, name)
        st.text_area(
            "Contenu extrait",
            value=ex.text,
            height=700,
            disabled=True,
            label_visibility="collapsed",
        )


def _skills_tags(skills: list, tone: str = "green") -> str:
    """Tags de compétences — style aplat, palette TriCV."""
    if not skills:
        return (
            "<span style='color:#8A8A86;font-style:italic;font-size:12.5px'>"
            "Aucune</span>"
        )

    if tone == "green":
        bg, fg, bd = "#EDF7F0", "#1A6E3C", "#CFE8D6"
    else:  # red
        bg, fg, bd = "#FCECEE", "#A30E25", "#F3CFD4"

    return " ".join(
        f"<span style='background:{bg};color:{fg};border:1px solid {bd};"
        f"padding:4px 11px;border-radius:999px;font-size:12px;font-weight:600;"
        f"margin:3px 3px 3px 0;display:inline-block;line-height:1.6'>{s}</span>"
        for s in skills
    )


def _render_llm_sections(text: str) -> None:
    """Affiche le texte LLM en sections séparées par des dividers."""
    sections = re.split(r"(?=### \d+\.)", text)
    for i, section in enumerate(sections):
        if section.strip():
            st.markdown(section)
            if i < len(sections) - 1:
                st.divider()


def render_result(result, hr_analysis: str = None, llm_error: str = None):
    skill_score_pct = round(result.scores.get("skill_score", result.scores["skill_coverage"]) * 100, 1)
    sem_display_pct = round(result.scores.get("semantic_display_score", result.scores["semantic_score"]) * 100, 1)

    n_matched = len(result.matched_skills)
    n_missing = len(result.missing_skills)
    n_total   = n_matched + n_missing

    # ── 1. Scores ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Adéquation compétences",
            f"{skill_score_pct}%",
            help=(
                "Proportion des compétences techniques demandées dans l'offre "
                "qui apparaissent dans le CV du candidat."
            ),
        )
        st.progress(skill_score_pct / 100)
        st.caption(f"{n_matched} / {n_total} compétences de l'offre présentes dans le CV")

    with col2:
        if sem_display_pct >= 70:
            sem_label = "Vocabulaire très aligné"
        elif sem_display_pct >= 50:
            sem_label = "Bonne proximité"
        elif sem_display_pct >= 30:
            sem_label = "Proximité partielle"
        else:
            sem_label = "Profils éloignés"

        st.metric(
            "Proximité sémantique",
            f"{sem_display_pct}%",
            help=(
                "Mesure à quel point le vocabulaire et les expériences du candidat "
                "ressemblent à ce que décrit l'offre, au-delà des mots-clés exacts."
            ),
        )
        st.progress(sem_display_pct / 100)
        st.caption(sem_label)

    st.divider()

    # ── 2. Compétences matchées / manquantes ───────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"**Compétences présentes** ({n_matched})")
        st.markdown(
            _skills_tags(result.matched_skills, tone="green"),
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(f"**Compétences absentes** ({n_missing})")
        st.markdown(
            _skills_tags(result.missing_skills, tone="red"),
            unsafe_allow_html=True,
        )

    if llm_error is not None:
        st.divider()
        st.subheader("Analyse RH")
        st.warning("Analyse RH indisponible. Vérifiez que le token LLM est bien configuré.")
    elif hr_analysis is not None:
        st.divider()
        st.subheader("Analyse RH")
        _render_llm_sections(hr_analysis)


# Valeurs par défaut (non exposées dans la sidebar)
skill_path = str(DATA_DIR / "skills_reference.json")
company_skill_path = str(DATA_DIR / "skills_company.json")
ollama_model = "llama3.2:3b"
emb_model_path = str(MODELS_DIR / "all-MiniLM-L6-v2") if MODELS_DIR.exists() else "models/all-MiniLM-L6-v2"

with st.sidebar:
    # Wordmark TriCV
    st.markdown(SIDEBAR_BRAND_HTML, unsafe_allow_html=True)

    st.subheader("Sémantique (embeddings)")
    semantic_threshold = st.slider("Seuil sémantique", 0.0, 1.0, 0.35, 0.01)
    breakpoint_percentile = st.slider(
        "Sensibilité chunking sémantique (%)",
        min_value=50, max_value=99, value=95, step=1,
        help="Plus la valeur est haute, plus les chunks sont petits et thématiquement homogènes.",
    )

    st.markdown("---")
    st.subheader("Scoring")
    verdict_threshold = st.slider("Seuil 'cohérent' (%)", 0, 100, 60, 5)


try:
    skill_ref = load_skill_ref(skill_path, company_skill_path)
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


tab1, tab3 = st.tabs(["Compare documents", "Classement multi-candidats"])

# ----------------------
# TAB 1: Documents
# ----------------------
_ACCEPTED_TYPES = ["pdf", "jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp", "docx", "txt"]

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        cv_pdf = st.file_uploader(
            "Upload CV (PDF, image, DOCX, TXT)",
            type=_ACCEPTED_TYPES,
            key="cv_pdf",
        )
    with col2:
        job_pdf = st.file_uploader(
            "Upload offre d'emploi (PDF, image, DOCX, TXT)",
            type=_ACCEPTED_TYPES,
            key="job_pdf",
        )

    # ── Extraction et prévisualisation de l'offre dès l'upload ───────────────
    _job_text_tab1 = None
    _job_label_tab1 = "OFFRE D'EMPLOI"

    if job_pdf:
        _job_ex = extract_from_bytes(job_pdf.getvalue(), job_pdf.name)
        if _job_ex.warning:
            st.warning(f"Offre — {_job_ex.warning}")
        if _job_ex.ocr_used:
            st.info("Offre : texte extrait par OCR (Tesseract) depuis une image.")

        if _job_ex.text.strip():
            _profil_text, _profil_found = extract_profil_section(_job_ex.text)
            _job_text_tab1 = _profil_text

            if _profil_found:
                _job_label_tab1 = "SECTION « VOTRE PROFIL » (exigences candidat)"
            else:
                st.warning(
                    "Section « Votre profil » non détectée dans l'offre — "
                    "le texte complet sera utilisé pour l'analyse."
                )

            with st.expander("Section analysée de l'offre", expanded=not _profil_found):
                if _profil_found:
                    st.caption(
                        "Section « Votre profil » extraite avec succès. "
                        "C'est ce texte — et uniquement lui — qui est transmis au moteur d'analyse."
                    )
                else:
                    st.caption(
                        "Texte intégral de l'offre (section « Votre profil » non détectée). "
                        "Vérifiez que le PDF contient bien un en-tête « Votre profil » sur sa propre ligne."
                    )
                st.text_area(
                    "Texte utilisé",
                    value=_profil_text,
                    height=260,
                    disabled=True,
                    key="profil_preview_tab1",
                )

    # ── Aperçu des documents uploadés (fermé par défaut) ─────────────────────
    if cv_pdf:
        with st.expander(f"Voir le CV — {cv_pdf.name}"):
            _render_document_preview(cv_pdf.name, cv_pdf.getvalue())
    if job_pdf:
        with st.expander(f"Voir l'offre — {job_pdf.name}"):
            _render_document_preview(job_pdf.name, job_pdf.getvalue())

    use_llm = st.checkbox("Générer une analyse RH", value=True, key="use_llm_tab1")
    run = st.button("Comparer", type="primary", use_container_width=True, key="run_pdf")

    if run:
        if not cv_pdf or not job_pdf:
            st.error("Il faut uploader les 2 documents.")
            st.stop()

        if not _job_text_tab1 or not _job_text_tab1.strip():
            st.error("Texte vide après extraction de l'offre. Vérifie que le document est lisible.")
            st.stop()

        with st.spinner("Analyse en cours..."):
            d = analyze_candidate(
                cv_bytes=cv_pdf.getvalue(),
                cv_filename=cv_pdf.name,
                job_text=_job_text_tab1,
                skill_ref=skill_ref,
                emb_model=emb_model,
                semantic_threshold=semantic_threshold,
                breakpoint_percentile=breakpoint_percentile,
                use_llm=use_llm,
                ollama_model=ollama_model,
                job_text_label=_job_label_tab1,
            )
        if d["cv_warning"]:
            st.warning(f"CV — {d['cv_warning']}")
        if d["cv_ocr_used"]:
            st.info("CV : texte extrait par OCR (Tesseract) depuis une image.")

        if not d["cv_text"].strip():
            st.error("Texte vide après extraction du CV. Vérifie que le document est lisible (pas une image trop floue).")
            st.stop()

        render_result(
            d["_result"],
            hr_analysis=d["llm_analysis"],
            llm_error=d.get("llm_error"),
        )


# ----------------------
# TAB 3: Multi-candidate ranking
# ----------------------
with tab3:
    st.caption(
        "Comparez plusieurs candidats sur une même offre. "
        "Les CVs sont analysés en séquence et classés par score décroissant."
    )

    # ── 1. Offre d'emploi ────────────────────────────────────────────────────
    st.markdown("#### 1. Offre d'emploi")
    job_multi = st.file_uploader(
        "Upload offre d'emploi (PDF, image, DOCX, TXT)",
        type=_ACCEPTED_TYPES,
        key="job_multi",
    )

    _job_text_tab3 = None
    _job_label_tab3 = "OFFRE D'EMPLOI"

    if job_multi:
        _job_ex_multi_pre = extract_from_bytes(job_multi.getvalue(), job_multi.name)
        if _job_ex_multi_pre.warning:
            st.warning(f"Offre — {_job_ex_multi_pre.warning}")
        if _job_ex_multi_pre.ocr_used:
            st.info("Offre : texte extrait par OCR (Tesseract) depuis une image.")

        if _job_ex_multi_pre.text.strip():
            _profil_text_m, _profil_found_m = extract_profil_section(_job_ex_multi_pre.text)
            _job_text_tab3 = _profil_text_m

            if _profil_found_m:
                _job_label_tab3 = "SECTION « VOTRE PROFIL » (exigences candidat)"
            else:
                st.warning(
                    "Section « Votre profil » non détectée dans l'offre — "
                    "le texte complet sera utilisé pour l'analyse."
                )

            with st.expander("Section analysée de l'offre", expanded=not _profil_found_m):
                if _profil_found_m:
                    st.caption(
                        "Section « Votre profil » extraite avec succès. "
                        "C'est ce texte — et uniquement lui — qui est transmis au moteur d'analyse."
                    )
                else:
                    st.caption(
                        "Texte intégral de l'offre (section « Votre profil » non détectée). "
                        "Vérifiez que le PDF contient bien un en-tête « Votre profil » sur sa propre ligne."
                    )
                st.text_area(
                    "Texte utilisé",
                    value=_profil_text_m,
                    height=260,
                    disabled=True,
                    key="profil_preview_tab3",
                )

    # ── Aperçu de l'offre uploadée ───────────────────────────────────────────
    if job_multi:
        with st.expander(f"Voir l'offre — {job_multi.name}"):
            _render_document_preview(job_multi.name, job_multi.getvalue())

    # ── 2. CVs des candidats ─────────────────────────────────────────────────
    st.markdown("#### 2. CVs des candidats")

    cv_files = st.file_uploader(
        "CVs des candidats — sélectionner plusieurs fichiers",
        type=_ACCEPTED_TYPES,
        accept_multiple_files=True,
        key="cvs_multi",
    )

    use_llm = st.checkbox("Générer une analyse RH", value=True, key="use_llm_tab3")
    run_multi = st.button(
        "Analyser tous les candidats",
        type="primary",
        use_container_width=True,
        key="run_multi",
    )

    # ── Analyse ─────────────────────────────────────────────────────────────
    if run_multi:
        if not job_multi:
            st.error("Il faut uploader l'offre d'emploi.")
            st.stop()
        if not cv_files:
            st.error("Il faut uploader au moins un CV.")
            st.stop()

        if not _job_text_tab3 or not _job_text_tab3.strip():
            st.error("Texte vide après extraction de l'offre. Vérifie que le document est lisible.")
            st.stop()

        progress_bar = st.progress(0, text="Initialisation…")
        results_list = []
        errors = []

        for i, cv_file in enumerate(cv_files):
            progress_bar.progress(
                i / len(cv_files),
                text=f"Analyse de {cv_file.name} ({i + 1}/{len(cv_files)})…",
            )
            try:
                d = analyze_candidate(
                    cv_bytes=cv_file.getvalue(),
                    cv_filename=cv_file.name,
                    job_text=_job_text_tab3,
                    skill_ref=skill_ref,
                    emb_model=emb_model,
                    semantic_threshold=semantic_threshold,
                    breakpoint_percentile=breakpoint_percentile,
                    use_llm=use_llm,
                    ollama_model=ollama_model,
                    job_text_label=_job_label_tab3,
                )
                if not d["cv_text"].strip():
                    errors.append(f"{cv_file.name} : texte vide après extraction, ignoré.")
                    continue
                results_list.append(d)
            except Exception as exc:
                errors.append(f"{cv_file.name} : erreur d'analyse — {exc}")

        progress_bar.progress(1.0, text="Analyse terminée.")

        for err in errors:
            st.warning(err)

        st.session_state["multi_results"] = sorted(
            results_list, key=lambda x: x["global_score"], reverse=True
        )
        st.session_state["multi_job_text"] = _job_text_tab3
        st.session_state["multi_cv_bytes"] = {f.name: f.getvalue() for f in cv_files}

    # ── Affichage des résultats ──────────────────────────────────────────────
    if st.session_state.get("multi_results"):
        ranked = st.session_state["multi_results"]
        _cv_bytes_map = st.session_state.get("multi_cv_bytes", {})
        st.divider()
        st.subheader(f"Classement — {len(ranked)} candidat(s)")

        # Tableau de classement
        table_rows = []
        for rank, d in enumerate(ranked, 1):
            score = d["global_score"]
            if score >= verdict_threshold:
                verdict = "Cohérent"
            elif score >= verdict_threshold * 0.75:
                verdict = "Partiel"
            else:
                verdict = "À risque"

            row = {
                "Rang": rank,
                "Candidat": d["candidate_name"],
                "Adéquation compétences (%)": d.get("skill_score", d["skill_coverage"]),
                "Proximité sémantique (%)": d.get("semantic_display_score", d["semantic_score"]),
                "Compétences matchées": len(d["matched_skills"]),
                "Compétences manquantes": len(d["missing_skills"]),
                "Verdict": verdict,
            }
            table_rows.append(row)

        df_rank = pd.DataFrame(table_rows)

        col_config = {
            "Rang": st.column_config.NumberColumn(width="small"),
            "Adéquation compétences (%)": st.column_config.ProgressColumn(
                "Adéquation compétences (%)", min_value=0, max_value=100, format="%.1f%%",
                help="Proportion des compétences techniques de l'offre présentes dans le CV.",
            ),
            "Proximité sémantique (%)": st.column_config.ProgressColumn(
                "Proximité sémantique (%)", min_value=0, max_value=100, format="%.1f%%",
                help="Alignement du vocabulaire et des expériences avec l'offre (embeddings + TF-IDF).",
            ),
            "Verdict": st.column_config.TextColumn(width="medium"),
        }

        st.dataframe(df_rank, use_container_width=True, hide_index=True, column_config=col_config)

        # Export JSON global
        export_all = []
        for rank, d in enumerate(ranked, 1):
            entry = {
                "rank": rank,
                "candidate_name": d["candidate_name"],
                "cv_filename": d["cv_filename"],
                "global_score": d["global_score"],
                "skill_score": d.get("skill_score", d["skill_coverage"]),
                "semantic_display_score": d.get("semantic_display_score", d["semantic_score"]),
                "semantic_coverage": d["semantic_coverage"],
                "semantic_score": d["semantic_score"],
                "skill_coverage": d["skill_coverage"],
                "matched_skills": d["matched_skills"],
                "missing_skills": d["missing_skills"],
                "semantic_gaps": [
                    {
                        "similarite": round(g.best_sim, 2),
                        "extrait_offre": g.job_snippet,
                        "meilleur_cv": g.best_cv_snippet,
                    }
                    for g in d["semantic_gaps"]
                ],
                "llm_analysis": d["llm_analysis"],
                "scores": d["scores"],
            }
            export_all.append(entry)

        st.download_button(
            "Télécharger tous les résultats (JSON)",
            data=json.dumps(export_all, indent=2, ensure_ascii=False),
            file_name="classement_candidats.json",
            mime="application/json",
            use_container_width=True,
            key="dl_multi_all",
        )

        # Détail par candidat (expanders, ordonnés par rang)
        st.divider()
        st.subheader("Détail par candidat")

        for rank, d in enumerate(ranked, 1):
            score = d["global_score"]
            skill_pct = d.get("skill_score", d["skill_coverage"])
            sem_pct = d.get("semantic_display_score", d["semantic_score"])
            label_icon = "🟢" if score >= verdict_threshold else ("🟡" if score >= verdict_threshold * 0.75 else "🔴")

            with st.expander(
                f"#{rank} — {d['candidate_name']}  ·  compétences {skill_pct}% · sémantique {sem_pct}% {label_icon}",
                expanded=(rank == 1),
            ):
                cv_name = d.get("cv_filename", "")
                if cv_name and cv_name in _cv_bytes_map:
                    _tab_results, _tab_cv = st.tabs(["Résultats", f"CV — {cv_name}"])
                    with _tab_results:
                        render_result(
                            d["_result"],
                            hr_analysis=d["llm_analysis"],
                            llm_error=d.get("llm_error"),
                        )
                    with _tab_cv:
                        _render_document_preview(cv_name, _cv_bytes_map[cv_name])
                else:
                    render_result(
                        d["_result"],
                        hr_analysis=d["llm_analysis"],
                        llm_error=d.get("llm_error"),
                    )
