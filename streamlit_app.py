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
from src.gestmax_parser import parse_gestmax_export, match_cv_to_candidate
from src.job_offer_parser import extract_profil_section


# Force offline pour HF/Transformers (si le modèle est local, aucun souci)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

st.set_page_config(
    page_title="Analyse CV / Offre — Outil RH",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Injection CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ───────────────────────────────────────────── */
.stApp {
    font-family: 'Arial', 'Helvetica Neue', sans-serif;
    background-color: #EDEAE5;
    background-image: radial-gradient(ellipse 70% 40% at 100% 0%,
        rgba(230, 51, 41, 0.05) 0%, transparent 60%);
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar : dégradé noir profond ──────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1A1815 0%, #252220 60%, #1A1815 100%) !important;
    border-right: 1px solid rgba(230, 51, 41, 0.2) !important;
}
[data-testid="stSidebar"] *, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; }
[data-testid="stSidebar"] input[type="text"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 4px !important;
    color: #FFFFFF !important;
}

/* ── Métriques : cartes sombres avec dégradé ─────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #282522 0%, #322F2B 100%);
    border: none;
    border-left: 3px solid #E63329;
    border-radius: 2px;
    padding: 1.4rem 1.8rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18), 0 2px 8px rgba(0,0,0,0.12);
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::after {
    content: "";
    position: absolute;
    top: 0; right: 0;
    width: 60%; height: 100%;
    background: radial-gradient(ellipse at top right,
        rgba(230, 51, 41, 0.08) 0%, transparent 70%);
    pointer-events: none;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #FFFFFF;
    font-size: 2.2rem;
    font-weight: 700;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.45);
    font-weight: 700;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: rgba(255,255,255,0.55);
    font-size: 0.82rem;
}

/* ── Boutons MBDA plein rouge ────────────────────────── */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #E63329 0%, #CC2D24 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 2px !important;
    font-weight: 700 !important;
    font-family: 'Arial', 'Helvetica Neue', sans-serif !important;
    padding: 0.6rem 2rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 0.78rem !important;
    box-shadow: 0 4px 16px rgba(230, 51, 41, 0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    box-shadow: 0 6px 24px rgba(230, 51, 41, 0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #E0E0E0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #999999;
    font-weight: 700;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 0.8rem 2rem;
    border: none;
    border-radius: 0;
    transition: color 0.15s;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #000000 !important;
    border-bottom: 3px solid #E63329 !important;
}

/* ── Expander ────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.7) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid #E0E0E0 !important;
    border-top: 2px solid #333333 !important;
    border-radius: 0 !important;
}
[data-testid="stExpander"] summary {
    color: #333333 !important;
    font-weight: 700;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Alertes contrastées ─────────────────────────────── */
div.stSuccess {
    background: linear-gradient(135deg, rgba(10,40,22,0.92) 0%, rgba(15,55,30,0.92) 100%) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(46,125,82,0.3) !important;
    border-left: 4px solid #2E7D52 !important;
    border-radius: 2px !important;
    color: #FFFFFF !important;
}
div.stWarning {
    background: linear-gradient(135deg, rgba(40,28,0,0.92) 0%, rgba(55,38,0,0.92) 100%) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(200,125,0,0.3) !important;
    border-left: 4px solid #C87D00 !important;
    border-radius: 2px !important;
    color: #FFFFFF !important;
}
div.stError {
    background: linear-gradient(135deg, rgba(40,8,6,0.92) 0%, rgba(60,12,10,0.92) 100%) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(230,51,41,0.3) !important;
    border-left: 4px solid #E63329 !important;
    border-radius: 2px !important;
    color: #FFFFFF !important;
}

/* ── Dataframe ───────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #E0E0E0;
    border-top: 2px solid #1A1A1A;
    border-radius: 0;
    background: rgba(255,255,255,0.8);
    backdrop-filter: blur(6px);
}

/* ── Séparateurs ─────────────────────────────────────── */
hr { border-color: #E0E0E0 !important; margin: 2rem 0; }

/* ── Titres h2 ───────────────────────────────────────── */
h2 {
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}
h2::before {
    content: "\25CF";
    color: #E63329;
    margin-right: 10px;
    font-size: 9px;
    vertical-align: middle;
}
h3 { color: #333333 !important; font-weight: 600 !important; }

/* ── Captions ────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {
    color: #888888 !important;
    font-size: 13px !important;
}
a { color: #E63329 !important; }
</style>
""", unsafe_allow_html=True)

# ── En-tête ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1C1A17 0%, #252220 50%, #2A1A18 100%);
    padding: 2.8rem 3rem 2.4rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
">
    <div style="
        position: absolute; top: -40px; right: -40px;
        width: 280px; height: 280px; border-radius: 50%;
        background: radial-gradient(circle, rgba(230,51,41,0.18) 0%, transparent 70%);
        pointer-events: none;
    "></div>
    <div style="
        position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #E63329 0%, rgba(230,51,41,0.2) 60%, transparent 100%);
    "></div>
    <div style="color: rgba(255,255,255,0.3); font-size: 11px; font-weight: 700;
                letter-spacing: 3px; text-transform: uppercase; margin-bottom: 1rem;">
        &#9679; &nbsp; Outil d'aide au recrutement &nbsp;&middot;&nbsp; Analyse offline
    </div>
    <div style="
        color: #FFFFFF;
        font-size: 2.6rem;
        font-weight: 300;
        font-family: 'Arial', 'Helvetica Neue', sans-serif;
        text-transform: uppercase;
        letter-spacing: 4px;
        line-height: 1.1;
    ">
        Analyse CV
        <span style="color: #E63329; font-weight: 700;">/</span>
        Offre d'emploi
    </div>
</div>
""", unsafe_allow_html=True)


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


def _skills_tags(skills: list, bg: str) -> str:
    if not skills:
        return "<span style='color:rgba(255,255,255,0.35);font-style:italic'>Aucune</span>"
    return " ".join(
        f"<span style='background:{bg};color:#fff;padding:3px 11px;border-radius:20px;"
        f"font-size:0.82em;margin:2px;display:inline-block;line-height:1.8'>{s}</span>"
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
            _skills_tags(result.matched_skills, "#1a6e3c"),
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(f"**Compétences absentes** ({n_missing})")
        st.markdown(
            _skills_tags(result.missing_skills, "#9b2222"),
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 3. Analyse RH ──────────────────────────────────────────────────────────
    st.subheader("Analyse RH")

    # ── DEBUG TEMPORAIRE — retirer après confirmation ─────────────────────────
    _dbg_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    _PLACEHOLDER = "your-mistral-api-key-here"
    if not _dbg_key:
        _key_status = "❌ absente"
    elif _dbg_key == _PLACEHOLDER:
        _key_status = f"❌ placeholder non remplacé ({len(_dbg_key)} chars)"
    elif len(_dbg_key) < 32:
        _key_status = f"❌ trop courte ({len(_dbg_key)} chars, min 32)"
    else:
        _key_status = f"✅ présente ({len(_dbg_key)} chars)"

    if llm_error:
        _analysis_status = f"❌ LLM a échoué — erreur : {llm_error}"
    elif hr_analysis and "Analyse LLM indisponible" in hr_analysis:
        _analysis_status = "⚠️ contenu = fallback statique (use_llm=False lors de l'analyse, ou analyse relancée sans LLM)"
    elif hr_analysis:
        _analysis_status = "✅ contenu LLM reçu"
    else:
        _analysis_status = "— aucune analyse"

    with st.expander("🔍 Debug LLM (temporaire)", expanded=True):
        st.write(f"**Clé API :** {_key_status}")
        st.write(f"**Analyse :** {_analysis_status}")
        if hr_analysis:
            st.write("**Contenu hr_analysis (300 premiers chars) :**")
            st.code(hr_analysis[:300], language=None)
    # ── FIN DEBUG ──────────────────────────────────────────────────────────────

    if llm_error is not None:
        st.error(f"**L'analyse LLM a échoué.** Raison : {llm_error}")
    else:
        _render_llm_sections(hr_analysis or "*(Aucune analyse disponible.)*")


with st.sidebar:
    st.header("Paramètres")
    skill_path = st.text_input(
        "Base de compétences générale",
        value=str(DATA_DIR / "skills_reference.json"),
    )
    company_skill_path = st.text_input(
        "Base de compétences entreprise (optionnel)",
        value=str(DATA_DIR / "skills_company.json"),
        help="Fichier JSON avec les compétences propres à votre entreprise. Laisser vide pour ne pas l'utiliser.",
    )

    st.markdown("---")
    st.subheader("Sémantique (embeddings)")
    # IMPORTANT: mets un chemin local vers un modèle déjà présent sur le disque
    default_model = str(MODELS_DIR / "all-MiniLM-L6-v2") if MODELS_DIR.exists() else "models/all-MiniLM-L6-v2"
    emb_model_path = st.text_input("Embedding model (local path)", value=default_model)
    semantic_threshold = st.slider("Seuil sémantique", 0.0, 1.0, 0.35, 0.01)
    breakpoint_percentile = st.slider(
        "Sensibilité chunking sémantique (%)",
        min_value=50, max_value=99, value=95, step=1,
        help="Plus la valeur est haute, plus les chunks sont petits et thématiquement homogènes.",
    )

    st.markdown("---")
    verdict_threshold = st.slider("Seuil 'cohérent' (%)", 0, 100, 60, 5)

    st.markdown("---")
    st.subheader("Analyse RH (Mistral API)")
    use_llm = st.checkbox("Générer une analyse RH (Mistral Small)", value=False)
    _mistral_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if use_llm:
        if _mistral_key:
            st.caption(f"Clé API : {'*' * 8}{_mistral_key[-4:]} · modèle : mistral-small-latest")
        else:
            st.error(
                "**MISTRAL_API_KEY manquante.** "
                "Ajoutez `MISTRAL_API_KEY=sk-...` dans le fichier `.env` "
                "à la racine du projet, puis relancez l'application.",
                icon="🔑",
            )
    # Ollama conservé pour rollback — réactiver si besoin
    ollama_model = "llama3.2:3b"   # non utilisé, gardé pour compatibilité des appels

    st.markdown("---")
    st.caption("Embeddings : offline. Analyse RH : Mistral API (cloud).")


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


tab1, tab2, tab3 = st.tabs(["Compare documents", "Demo dataset", "Classement multi-candidats"])

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
    # Effectuée ici (hors du bloc "if run") pour afficher l'expander
    # immédiatement et réutiliser le résultat sans double extraction.
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
        with st.spinner("Analyse en cours..."):
            d = analyze_candidate(
                cv_bytes=cv_text.encode("utf-8"),
                cv_filename="candidate.txt",
                job_text=job_text,
                skill_ref=skill_ref,
                emb_model=emb_model,
                semantic_threshold=semantic_threshold,
                breakpoint_percentile=breakpoint_percentile,
                use_llm=use_llm,
                ollama_model=ollama_model,
            )
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

    # ── 2. Import Gestmax (optionnel) ────────────────────────────────────────
    st.markdown("#### 2. Import Gestmax *(optionnel)*")
    st.info(
        "**Gestmax** : exportez la liste des candidatures depuis "
        "*Gestion > Candidatures > Exporter* (format CSV ou Excel). "
        "L'export doit contenir au minimum une colonne nom/prénom. "
        "Les colonnes date de candidature et statut sont exploitées automatiquement si présentes. "
        "Les CVs uploadés ci-dessous seront rapprochés des candidats par correspondance de nom.",
        icon="ℹ️",
    )

    gestmax_file = st.file_uploader(
        "Export Gestmax (.csv ou .xlsx)",
        type=["csv", "xlsx", "xls"],
        key="gestmax_file",
    )

    # Parsing et preview Gestmax
    if gestmax_file:
        gx_result = parse_gestmax_export(gestmax_file.getvalue(), gestmax_file.name)

        for w in gx_result.warnings:
            st.warning(w)

        if gx_result.candidates:
            st.session_state["gestmax_candidates"] = gx_result.candidates

            # Colonnes détectées
            if gx_result.detected_columns:
                detected_str = " · ".join(
                    f"**{role}** → `{col}`"
                    for role, col in gx_result.detected_columns.items()
                )
                st.caption(f"Colonnes détectées : {detected_str}")

            # Aperçu du tableau Gestmax
            preview_rows = []
            for c in gx_result.candidates[:50]:
                preview_rows.append({
                    "Candidat (Gestmax)": c.candidate_name,
                    "Date candidature": c.application_date or "—",
                    "Statut": c.status or "—",
                    "Offre / Poste": c.offer_ref or "—",
                    "Fichier CV (export)": c.cv_filename_hint or "—",
                })
            df_preview = pd.DataFrame(preview_rows)
            with st.expander(
                f"Aperçu import Gestmax — {len(gx_result.candidates)} candidat(s)",
                expanded=True,
            ):
                st.dataframe(df_preview, use_container_width=True, hide_index=True)
                if len(gx_result.candidates) > 50:
                    st.caption(f"… et {len(gx_result.candidates) - 50} autres candidats non affichés.")
        else:
            # Fichier uploadé mais rien extrait — ne pas garder d'ancien état
            st.session_state.pop("gestmax_candidates", None)
    elif not gestmax_file:
        # Pas de fichier : on efface les candidats Gestmax précédents pour éviter
        # qu'un ancien import soit réutilisé silencieusement après suppression du fichier.
        st.session_state.pop("gestmax_candidates", None)

    # ── 3. CVs des candidats ─────────────────────────────────────────────────
    st.markdown("#### 3. CVs des candidats")

    gestmax_candidates = st.session_state.get("gestmax_candidates", [])
    if gestmax_candidates:
        st.caption(
            f"{len(gestmax_candidates)} candidat(s) importés depuis Gestmax. "
            "Uploadez leurs CVs ci-dessous — le rapprochement se fait automatiquement par nom de fichier."
        )

    cv_files = st.file_uploader(
        "CVs des candidats — sélectionner plusieurs fichiers",
        type=_ACCEPTED_TYPES,
        accept_multiple_files=True,
        key="cvs_multi",
    )

    # Aperçu du matching Gestmax ↔ fichiers uploadés
    if cv_files and gestmax_candidates:
        match_rows = []
        for f in cv_files:
            hit = match_cv_to_candidate(f.name, gestmax_candidates)
            match_rows.append({
                "Fichier uploadé": f.name,
                "Candidat Gestmax associé": hit.candidate_name if hit else "⚠ Non trouvé",
                "Date candidature": hit.application_date if hit else "—",
            })
        with st.expander("Vérifier le rapprochement Gestmax ↔ CVs uploadés", expanded=False):
            st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)
            unmatched = sum(1 for r in match_rows if r["Candidat Gestmax associé"].startswith("⚠"))
            if unmatched:
                st.caption(
                    f"{unmatched} fichier(s) non associé(s) — leurs noms seront utilisés tels quels. "
                    "Renommez les CVs au format «Prénom Nom» pour améliorer la détection."
                )

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

        # Snapshot des candidats Gestmax pour cette analyse
        gx_candidates_snap = st.session_state.get("gestmax_candidates", [])

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

                # Enrichissement avec les métadonnées Gestmax si disponibles
                gx_match = match_cv_to_candidate(cv_file.name, gx_candidates_snap) if gx_candidates_snap else None
                d["gestmax_name"] = gx_match.candidate_name if gx_match else None
                d["gestmax_date"] = gx_match.application_date if gx_match else None
                d["gestmax_status"] = gx_match.status if gx_match else None
                d["gestmax_offer"] = gx_match.offer_ref if gx_match else None
                d["gestmax_matched"] = gx_match is not None

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

    # ── Affichage des résultats ──────────────────────────────────────────────
    if st.session_state.get("multi_results"):
        ranked = st.session_state["multi_results"]
        job_text_multi = st.session_state.get("multi_job_text", "")
        has_gestmax = any(d.get("gestmax_matched") for d in ranked)

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

            # Nom affiché : priorité au nom Gestmax, fallback filename stem
            display_name = d["gestmax_name"] if d.get("gestmax_name") else d["candidate_name"]

            row = {
                "Rang": rank,
                "Candidat": display_name,
                "Adéquation compétences (%)": d.get("skill_score", d["skill_coverage"]),
                "Proximité sémantique (%)": d.get("semantic_display_score", d["semantic_score"]),
                "Compétences matchées": len(d["matched_skills"]),
                "Compétences manquantes": len(d["missing_skills"]),
                "Verdict": verdict,
            }
            if has_gestmax:
                row["Date candidature"] = d.get("gestmax_date") or "—"
                row["Statut Gestmax"] = d.get("gestmax_status") or "—"

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
        if has_gestmax:
            col_config["Date candidature"] = st.column_config.TextColumn(width="medium")
            col_config["Statut Gestmax"] = st.column_config.TextColumn(width="medium")

        st.dataframe(df_rank, use_container_width=True, hide_index=True, column_config=col_config)

        # Export JSON global
        export_all = []
        for rank, d in enumerate(ranked, 1):
            entry = {
                "rank": rank,
                "candidate_name": d["gestmax_name"] if d.get("gestmax_name") else d["candidate_name"],
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
            if d.get("gestmax_matched"):
                entry["gestmax"] = {
                    "nom": d["gestmax_name"],
                    "date_candidature": d["gestmax_date"],
                    "statut": d["gestmax_status"],
                    "offre": d["gestmax_offer"],
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
            display_name = d["gestmax_name"] if d.get("gestmax_name") else d["candidate_name"]
            label_icon = "🟢" if score >= verdict_threshold else ("🟡" if score >= verdict_threshold * 0.75 else "🔴")

            with st.expander(
                f"#{rank} — {display_name}  ·  compétences {skill_pct}% · sémantique {sem_pct}% {label_icon}",
                expanded=(rank == 1),
            ):
                # Bandeau métadonnées Gestmax si disponible
                if d.get("gestmax_matched"):
                    meta_parts = []
                    if d["gestmax_date"]:
                        meta_parts.append(f"Candidature : **{d['gestmax_date']}**")
                    if d["gestmax_status"]:
                        meta_parts.append(f"Statut : **{d['gestmax_status']}**")
                    if d["gestmax_offer"]:
                        meta_parts.append(f"Poste : **{d['gestmax_offer']}**")
                    if meta_parts:
                        st.caption("  ·  ".join(meta_parts))
                    st.caption(f"Fichier CV : `{d['cv_filename']}`")

                render_result(
                    d["_result"],
                    hr_analysis=d["llm_analysis"],
                    llm_error=d.get("llm_error"),
                )
