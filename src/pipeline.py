"""
Pipeline d'analyse candidat — point d'entrée unique pour la logique métier.

Usage:
    from src.pipeline import analyze_candidate

    result = analyze_candidate(
        cv_bytes=file.getvalue(),
        cv_filename=file.name,
        job_text=job_text,
        skill_ref=skill_ref,
        emb_model=emb_model,
    )
    print(result["global_score"], result["missing_skills"], result["llm_analysis"])
"""

from __future__ import annotations

import os
import requests
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.comparator import compare_texts
from src.doc_reader import extract_from_bytes, ExtractedDoc
from src.skills_reference import SkillReference

# Charge les variables d'env depuis .env s'il existe (sans écraser les vars d'env existantes)
load_dotenv(override=False)


# ── Utilitaire texte ──────────────────────────────────────────────────────────

def _clip(text: str, max_chars: int = 6000) -> str:
    return (text or "")[:max_chars]


# ── Abstraction LLM (Mistral public ou chatbot interne MBDA) ─────────────────

_llm_call_count: int = 0  # compteur de session, remis à zéro par reset_llm_counter()


def reset_llm_counter() -> None:
    global _llm_call_count
    _llm_call_count = 0


def get_llm_call_count() -> int:
    return _llm_call_count


def call_llm(prompt: str) -> str:
    """
    Appelle le fournisseur LLM configuré via LLM_PROVIDER.

    Returns:
        Texte généré.
    Raises:
        ValueError  : configuration manquante ou invalide.
        Exception   : toute erreur réseau / API (propagée telle quelle).
    """
    global _llm_call_count
    _llm_call_count += 1
    print(f"[LLM CALL #{_llm_call_count}] prompt length: {len(prompt)} chars", flush=True)

    provider = os.getenv("LLM_PROVIDER", "mistral").strip().lower()

    if provider == "internal":
        url   = os.getenv("INTERNAL_LLM_URL", "").strip()
        token = os.getenv("INTERNAL_LLM_TOKEN", "").strip()
        model = os.getenv("INTERNAL_LLM_MODEL", "Mistral-Small-3.1").strip()

        if not url or not token:
            raise ValueError(
                "INTERNAL_LLM_URL ou INTERNAL_LLM_TOKEN manquant dans .env — "
                "renseignez ces deux variables pour utiliser le chatbot interne MBDA."
            )

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=60,
            verify=False,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    else:  # mistral public API
        _PLACEHOLDER = "your-mistral-api-key-here"
        api_key = os.environ.get("MISTRAL_API_KEY", "").strip()

        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY manquante — "
                "ajoutez MISTRAL_API_KEY=sk-... dans le fichier .env et relancez l'application."
            )
        if api_key == _PLACEHOLDER:
            raise ValueError(
                f"MISTRAL_API_KEY contient la valeur placeholder du template (\"{_PLACEHOLDER}\"). "
                "Remplacez-la par votre vraie clé sur https://console.mistral.ai/api-keys"
            )
        if len(api_key) < 32:
            raise ValueError(
                f"MISTRAL_API_KEY semble invalide : longueur {len(api_key)} caractères "
                f"(valeur actuelle : \"{api_key[:6]}…\"). "
                "Une clé Mistral valide fait au moins 32 caractères."
            )

        from mistralai import Mistral
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""


# ── LLM local (Ollama) — conservé pour rollback ───────────────────────────────

# def run_ollama(model: str, prompt: str, timeout_s: int = 90) -> tuple[Optional[str], Optional[str]]:
#     try:
#         p = subprocess.run(
#             ["ollama", "run", model, prompt],
#             text=True,
#             capture_output=True,
#             timeout=timeout_s,
#         )
#         if p.returncode != 0:
#             return None, (p.stderr or "").strip() or f"Ollama exited with {p.returncode}"
#         out = (p.stdout or "").strip()
#         return out if out else None, None
#     except FileNotFoundError:
#         return None, "Ollama n'est pas installé (commande 'ollama' introuvable)."
#     except subprocess.TimeoutExpired:
#         return None, "Timeout Ollama (prompt trop long ou modèle lent)."


# ── LLM Mistral API ───────────────────────────────────────────────────────────

MISTRAL_MODEL = "mistral-small-latest"


def run_mistral(prompt: str, temperature: float = 0.3) -> tuple[Optional[str], Optional[str]]:
    """
    Appelle l'API Mistral (api.mistral.ai) avec le modèle mistral-small-latest.

    Returns:
        (output, error) — l'un des deux est None.
        - output : texte généré si succès
        - error  : message d'erreur lisible si échec
    """
    import traceback

    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()

    # ── DEBUG (retirer après confirmation) ────────────────────────────────────
    print(f"[Mistral DEBUG] api_key présente: {bool(api_key)}, longueur: {len(api_key)}", flush=True)
    print(f"[Mistral DEBUG] prompt (500 premiers chars):\n{prompt[:500]}\n---", flush=True)
    # ── FIN DEBUG ──────────────────────────────────────────────────────────────

    _PLACEHOLDER = "your-mistral-api-key-here"

    if not api_key:
        return None, (
            "MISTRAL_API_KEY manquante — "
            "ajoutez MISTRAL_API_KEY=sk-... dans le fichier .env et relancez l'application."
        )
    if api_key == _PLACEHOLDER:
        return None, (
            "MISTRAL_API_KEY contient la valeur placeholder du template "
            f"(\"{_PLACEHOLDER}\"). "
            "Remplacez-la par votre vraie clé sur https://console.mistral.ai/api-keys"
        )
    if len(api_key) < 32:
        return None, (
            f"MISTRAL_API_KEY semble invalide : longueur {len(api_key)} caractères "
            f"(valeur actuelle : \"{api_key[:6]}…\"). "
            "Une clé Mistral valide fait au moins 32 caractères."
        )

    try:
        from mistralai import Mistral
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return (content.strip() if content else None), None
    except ImportError:
        return None, "Package 'mistralai' non installé. Lancez : pip install mistralai"
    except Exception as exc:
        traceback.print_exc()
        return None, f"{type(exc).__name__}: {exc}"


# ── Analyse RH sans LLM (fallback) ───────────────────────────────────────────

def fallback_analysis(missing_skills: list[str], gaps_snippets: list[str], score_pct: float) -> str:
    """Analyse RH de fallback quand le LLM est indisponible. Structure alignée sur le prompt LLM."""
    if score_pct >= 70:
        decision = "CONVOQUER"
        just = "Le score global indique que le CV couvre les grandes exigences de l'offre."
    elif score_pct >= 50:
        decision = "À ÉTUDIER"
        just = "Le CV répond partiellement aux exigences — à examiner si le vivier est limité."
    else:
        decision = "NE PAS CONVOQUER"
        just = "Le CV ne couvre pas suffisamment les exigences principales de l'offre."

    lines = [
        f"### 1. DÉCISION\n**{decision}**\n\n{just}",
        "### 2. EXPLICATION DES SCORES\n*(Analyse LLM indisponible — les scores bruts sont visibles ci-dessus.)*",
        "### 3. CE QUI PLAIDE POUR CE CANDIDAT\n*(Analyse LLM indisponible.)*",
    ]

    vigilance = ["### 4. POINTS DE VIGILANCE"]
    if missing_skills:
        for s in missing_skills[:4]:
            vigilance.append(f"- 🔴 **{s}** → compétence demandée dans l'offre, absente du CV")
    else:
        vigilance.append("*(Aucune lacune majeure détectée sur les compétences référencées.)*")
    lines.append("\n".join(vigilance))

    questions = ["### 5. QUESTIONS POUR L'ENTRETIEN"]
    if missing_skills:
        questions.append(
            f"- Vous n'avez pas mentionné **{missing_skills[0]}** dans votre CV. "
            "Avez-vous une expérience concrète avec cet outil ou cette technologie, "
            "et dans quel contexte l'avez-vous utilisé ?"
        )
    if len(missing_skills) > 1:
        questions.append(
            f"- **{missing_skills[1]}** est demandé dans l'offre mais absent de votre parcours. "
            "Comment avez-vous compensé cette absence dans vos missions précédentes ?"
        )
    if gaps_snippets:
        questions.append(
            f"- L'offre mentionne : *\"{gaps_snippets[0][:120]}\"*. "
            "Comment positionneriez-vous votre expérience par rapport à cette exigence ?"
        )
    if len(questions) == 1:
        questions.append("*(Analyse LLM indisponible — générez une analyse RH pour obtenir des questions ciblées.)*")
    lines.append("\n".join(questions))

    return "\n\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _semantic_label(pct: float) -> str:
    if pct >= 70:
        return "Vocabulaire très aligné"
    if pct >= 50:
        return "Bonne proximité"
    if pct >= 30:
        return "Proximité partielle"
    return "Profils éloignés"


# ── Analyse RH via LLM ────────────────────────────────────────────────────────

def generate_hr_analysis_llm(
    cv_text: str,
    job_text: str,
    matched_skills: list[str],
    missing_skills: list[str],
    semantic_gaps_snippets: list[str],
    score_pct: float,
    skill_score_pct: float,
    semantic_display_pct: float,
) -> tuple[Optional[str], Optional[str]]:
    matched_str  = ", ".join(matched_skills[:30])  if matched_skills  else "(aucune détectée)"
    missing_str  = ", ".join(missing_skills[:30])  if missing_skills  else "(aucune)"
    sem_label    = _semantic_label(semantic_display_pct)

    prompt = f"""You are an expert technical recruiter writing for an HR professional who is not technical. Your analysis must be concrete, nuanced, and based strictly on what is actually written in the CV and the offer. Never invent, assume, or generalize.

## EXIGENCES DU POSTE
{_clip(job_text, 3000)}

## CV DU CANDIDAT
{_clip(cv_text, 5000)}

## MÉTRIQUES CALCULÉES
- Compétences trouvées dans le CV : {matched_str} ({skill_score_pct}% de couverture)
- Compétences requises mais absentes : {missing_str}
- Proximité sémantique : {semantic_display_pct}% — {sem_label}
- Score global (classement interne) : {score_pct}%

---

Réponds en français. Utilise exactement les titres de section ci-dessous.

### 1. DÉCISION
Indique exactement l'une des options : **CONVOQUER** / **À ÉTUDIER** / **NE PAS CONVOQUER**

Puis écris 2-3 phrases expliquant la décision en te basant sur le profil réel. Sois direct. Si une lacune est critique (compétence cœur absente, niveau d'expérience insuffisant), nomme-la explicitement. Si le niveau d'expérience requis ne correspond pas à ce que montre le CV, signale-le avec ⚠️.

### 2. EXPLICATION DES SCORES
Cette section est destinée au recruteur RH qui voit des chiffres mais ne sait pas ce qu'ils signifient. Explique en français courant :
- Pourquoi ce candidat a-t-il obtenu ce score de compétences ? Quelles absences pèsent le plus, et sont-elles centrales au rôle ou périphériques ?
- Pourquoi a-t-il obtenu ce score sémantique ? Que révèle-t-il sur le profil — le candidat travaille-t-il dans un domaine similaire, un environnement tech similaire, un type de rôle similaire ?
- Les deux scores sont-ils cohérents entre eux, ou y a-t-il un écart (ex. bonne sémantique mais faibles compétences = quelqu'un du bon secteur mais qui manque d'outils spécifiques) ?
Écris 4-6 phrases. Pas de bullet points dans cette section — explication fluide.

### 3. CE QUI PLAIDE POUR CE CANDIDAT
Liste 2-3 points forts genuins pertinents pour CETTE offre spécifiquement. Pour chaque point, cite l'expérience ou le projet précis du CV qui l'étaye et explique pourquoi c'est utile pour ce rôle. Si aucun point fort n'est réellement pertinent pour ce poste, écris exactement : "Aucun point fort directement pertinent pour ce poste n'a été identifié." N'invente pas de points forts par politesse.

### 4. POINTS DE VIGILANCE
Liste 2-4 points que le recruteur devrait approfondir ou vérifier. Format pour chaque point : [lacune ou incertitude] → [pourquoi c'est important pour ce rôle] Distingue :
- 🔴 Bloquant : absence qui empêcherait de faire le travail
- 🟡 À vérifier : écart potentiellement comblable, à explorer en entretien
Ne recopie pas la liste de compétences — ne signale que ce qui compte vraiment dans le contexte de cette offre.

### 5. QUESTIONS POUR L'ENTRETIEN
Écris 2-3 questions qui aideraient le recruteur à décider pour CE candidat spécifique. Chaque question doit : faire référence à quelque chose de concret dans le CV de CE candidat (un poste, projet ou technologie qu'il a mentionné), cibler une incertitude réelle sur son adéquation, être posable en 5 minutes de conversation.

Mauvais exemple : "Avez-vous utilisé Kubernetes ?"
Bon exemple : "Dans votre rôle chez [entreprise mentionnée dans le CV], vous mentionnez avoir géré des déploiements — sur quelle infrastructure reposaient-ils, et aviez-vous la main sur la configuration des environnements ?"
""".strip()

    # ── Appel LLM (provider configuré via LLM_PROVIDER) ───────────────────────
    try:
        import traceback
        out = call_llm(prompt)
        return out, None
    except Exception as exc:
        traceback.print_exc()
        return None, f"{type(exc).__name__}: {exc}"


# ── Pipeline principal ────────────────────────────────────────────────────────

def analyze_candidate(
    cv_bytes: bytes,
    cv_filename: str,
    job_text: str,
    skill_ref: SkillReference,
    emb_model: SentenceTransformer,
    semantic_threshold: float = 0.35,
    breakpoint_percentile: int = 95,
    use_llm: bool = False,
    ollama_model: str = "llama3.2:3b",
    job_text_label: str = "OFFRE D'EMPLOI",
) -> dict:
    """
    Analyse complète d'un candidat par rapport à une offre d'emploi.

    Args:
        cv_bytes:             Contenu brut du fichier CV (PDF, image, DOCX, TXT).
        cv_filename:          Nom du fichier CV, utilisé pour détecter le format.
        job_text:             Texte de l'offre d'emploi (déjà extrait).
        skill_ref:            Référentiel de compétences chargé.
        emb_model:            Modèle d'embeddings local.
        semantic_threshold:   Seuil de similarité cosinus pour les gaps sémantiques.
        breakpoint_percentile: Sensibilité du chunking sémantique (50–99).
        use_llm:              Si True, génère l'analyse via Ollama.
        ollama_model:         Nom du modèle Ollama à utiliser.

    Returns:
        dict avec les clés :
          candidate_name, cv_filename, cv_source_type, cv_ocr_used, cv_warning,
          global_score, semantic_score, semantic_coverage, skill_coverage,
          matched_skills, missing_skills, semantic_gaps,
          llm_analysis, scores, _result (CompareResult, pour render_result).
    """
    # 1. Extraction du texte du CV
    cv_doc: ExtractedDoc = extract_from_bytes(cv_bytes, cv_filename)

    # 2. Comparaison sémantique + skills
    result = compare_texts(
        cv_text=cv_doc.text,
        job_text=job_text,
        skill_ref=skill_ref,
        emb_model=emb_model,
        semantic_threshold=semantic_threshold,
        breakpoint_percentile=breakpoint_percentile,
    )

    # 3. Analyse RH (LLM ou fallback)
    score_pct = result.scores["score_pct"]
    semantic_gaps_snippets = [g.job_snippet for g in result.semantic_gaps]

    if use_llm:
        llm_analysis, llm_error = generate_hr_analysis_llm(
            cv_text=cv_doc.text,
            job_text=job_text,
            matched_skills=result.matched_skills,
            missing_skills=result.missing_skills,
            semantic_gaps_snippets=semantic_gaps_snippets,
            score_pct=score_pct,
            skill_score_pct=round(result.scores["skill_score"] * 100, 1),
            semantic_display_pct=round(result.scores["semantic_display_score"] * 100, 1),
        )
    else:
        llm_analysis = fallback_analysis(
            missing_skills=result.missing_skills,
            gaps_snippets=semantic_gaps_snippets,
            score_pct=score_pct,
        )
        llm_error = None

    return {
        # Identification
        "candidate_name": Path(cv_filename).stem,
        "cv_filename": cv_filename,
        "cv_source_type": cv_doc.source_type,
        "cv_ocr_used": cv_doc.ocr_used,
        "cv_warning": cv_doc.warning,
        "cv_text": cv_doc.text,
        # Scores synthétiques (prêts à l'affichage)
        "global_score": score_pct,
        "skill_score": round(result.scores["skill_score"] * 100, 1),
        "semantic_display_score": round(result.scores["semantic_display_score"] * 100, 1),
        "semantic_score": round(result.scores["semantic_score"] * 100, 1),
        "semantic_coverage": round(result.scores["semantic_coverage"] * 100, 1),
        "skill_coverage": round(result.scores["skill_coverage"] * 100, 1),
        # Compétences
        "matched_skills": result.matched_skills,
        "missing_skills": result.missing_skills,
        # Détail sémantique
        "semantic_gaps": result.semantic_gaps,
        # Analyse RH
        "llm_analysis": llm_analysis,   # str si succès ou use_llm=False, None si API a échoué
        "llm_error": llm_error,          # str si API a échoué, None sinon
        # Scores bruts (pour export / détail technique)
        "scores": result.scores,
        # Objet CompareResult complet (pour render_result)
        "_result": result,
    }
