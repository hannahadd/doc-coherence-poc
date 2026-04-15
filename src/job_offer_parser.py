"""
Extraction de la section pertinente d'une offre d'emploi MBDA.

Les offres MBDA sont structurées en sections dont une seule — « Votre profil » —
contient les véritables exigences candidat. Tout le reste (présentation du poste,
avantages, localisation) est du bruit pour l'analyse de cohérence CV/offre.

La détection fonctionne sur du texte continu (PDF sans sauts de ligne) comme sur
du texte structuré avec sauts de ligne, grâce au flag re.DOTALL.

Usage :
    from src.job_offer_parser import extract_profil_section

    job_text, found = extract_profil_section(raw_text)
    if not found:
        logger.warning("Section 'Votre profil' non détectée — texte complet utilisé.")
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ── Normalisation ─────────────────────────────────────────────────────────────

def _strip_accents(s: str) -> str:
    """Supprime les accents (décomposition NFD) pour la comparaison robuste."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


# ── Patterns sur le texte ORIGINAL (avec accents) ────────────────────────────

_PROFIL_RE = re.compile(
    r"votre\s+profil\s*:?",
    re.IGNORECASE,
)

# En-têtes MBDA qui marquent la FIN de "Votre profil".
# Les variantes accentuées ([eé]) couvrent les deux encodages courants.
_END_SECTION_RE = re.compile(
    r"(?:"
    r"r[eé]tributions?\s+et\s+avantages?"   # Rétributions et avantages
    r"|r[eé]tribution\b"                     # Rétribution (forme courte)
    r"|avantages?\s+et\s+r[eé]tributions?"  # ordre inversé
    r"|les?\s*\+\s*de\s*l[\u2019\x27 ]offre"  # Les + de l'offre (apostrophes)
    r"|les?\s+plus\s+de\s*l[\u2019\x27 ]offre"
    r"|votre\s+quotidien"
    r"|votre\s+environnement"
    r"|le\s+site\b"
    r"|notre\s+site\b"
    r"|postuler\b"                           # bouton fin d'offre
    r"|nos?\s+engagements?"
    r"|informations?\s+compl[eé]mentaires?"
    r"|conditions?\s+de\s+travail"
    r")",
    re.IGNORECASE,
)

# ── Patterns sur le texte NORMALISÉ (sans accents) ───────────────────────────
# Utilisés quand le PDF a perdu les accents à l'extraction.

_PROFIL_RE_NORM = _PROFIL_RE   # "profil" n'a pas d'accent, même pattern

_END_SECTION_RE_NORM = re.compile(
    r"(?:"
    r"retributions?\s+et\s+avantages?"
    r"|retribution\b"
    r"|avantages?\s+et\s+retributions?"
    r"|les?\s*\+\s*de\s*l.offre"
    r"|les?\s+plus\s+de\s*l.offre"
    r"|votre\s+quotidien"
    r"|votre\s+environnement"
    r"|le\s+site\b"
    r"|notre\s+site\b"
    r"|postuler\b"
    r"|nos?\s+engagements?"
    r"|informations?\s+complementaires?"
    r"|conditions?\s+de\s+travail"
    r")",
    re.IGNORECASE,
)


# ── Extraction ────────────────────────────────────────────────────────────────

def _try_extract(
    search_text: str,
    source_text: str,
    profil_re: re.Pattern,
    end_re: re.Pattern,
) -> Optional[str]:
    """
    Cherche le début et la fin de la section dans search_text,
    puis extrait la tranche correspondante de source_text.

    Safe uniquement si len(search_text) == len(source_text) — ce qui est
    garanti quand search_text est _strip_accents(source_text) sur du texte
    NFC français standard (chaque précomposé → base seul, longueur identique).
    """
    profil_match = profil_re.search(search_text)
    if not profil_match:
        return None

    # Sauter le titre lui-même + éventuel ":" et espaces résiduels
    raw_start = profil_match.end()
    skip = re.match(r"[\s:]*", search_text[raw_start:])
    content_start = raw_start + (skip.end() if skip else 0)

    # Trouver le prochain en-tête de section
    end_match = end_re.search(search_text, pos=content_start)
    content_end = end_match.start() if end_match else len(search_text)

    # Extraire depuis source_text (même positions si len identique)
    extract_from = source_text if len(search_text) == len(source_text) else search_text
    section = extract_from[content_start:content_end].strip()

    return section if section else None


def extract_profil_section(text: str) -> tuple[str, bool]:
    """
    Extrait la section « Votre profil » d'une offre d'emploi MBDA.

    Fonctionne sur du texte continu (PDF sans sauts de ligne) ET sur du texte
    structuré (avec \\n entre les sections) grâce au matching sans ancre de ligne.

    Deux passes :
    1. Recherche directe sur le texte original (accents conservés).
    2. Recherche sur le texte normalisé sans accents (fallback si le PDF a perdu
       les diacritiques à l'extraction).

    Returns:
        (section_text, found)
        - section_text : contenu de « Votre profil » si trouvé, texte complet sinon
        - found        : True si la section a été localisée
    """
    if not text or not text.strip():
        return text, False

    # Passe 1 : texte original
    result = _try_extract(text, text, _PROFIL_RE, _END_SECTION_RE)
    if result:
        return result, True

    # Passe 2 : texte normalisé (PDF sans accents)
    norm = _strip_accents(text)
    result = _try_extract(norm, text, _PROFIL_RE_NORM, _END_SECTION_RE_NORM)
    if result:
        return result, True

    return text, False
