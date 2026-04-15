"""
Parser pour les exports Gestmax (CSV ou Excel).

Gestmax est un logiciel ATS utilisé dans les entreprises françaises.
Son format d'export n'est pas standardisé, mais ce module gère les noms
de colonnes les plus courants et reconstruit une liste de candidats.

Usage :
    from src.gestmax_parser import parse_gestmax_export, match_cv_to_candidate
    result = parse_gestmax_export(file_bytes, "export.xlsx")
    candidate = match_cv_to_candidate("dupont_jean_cv.pdf", result.candidates)
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


# ── Patterns de colonnes attendus dans les exports Gestmax ───────────────────
# Chaque liste est parcourue dans l'ordre ; le premier match gagne.

_FULLNAME_PATTERNS = [
    "nom prénom", "nom complet", "nom du candidat", "candidat",
    "prenom nom", "prénom nom", "nom_prenom", "civilite nom prenom",
]
_LASTNAME_PATTERNS = [
    "nom", "nom de famille", "last name", "lastname",
]
_FIRSTNAME_PATTERNS = [
    "prénom", "prenom", "first name", "firstname",
]
_DATE_PATTERNS = [
    "date de candidature", "date candidature", "date de dépôt",
    "date depot", "date dépôt", "date d'envoi", "date d envoi",
    "date de soumission", "date soumission", "date postulation",
    "date", "dépôt le", "postuléle",
]
_CVFILE_PATTERNS = [
    "cv", "fichier cv", "nom cv", "nom du cv", "pièce jointe cv",
    "piece jointe cv", "cv joint", "cv attaché", "cv attache",
    "fichier", "document", "attachment", "pièce jointe", "piece jointe",
]
_OFFER_PATTERNS = [
    "offre", "poste", "intitulé du poste", "intitule du poste",
    "référence offre", "reference offre", "référence", "reference",
    "intitulé", "intitule", "emploi",
]
_STATUS_PATTERNS = [
    "statut", "état", "etape", "étape", "status",
    "phase", "avancement", "décision", "decision",
]


# ── Normalisation ─────────────────────────────────────────────────────────────

def _norm_col(s: str) -> str:
    """Minuscules, sans accents, espaces collapsés."""
    s = unicodedata.normalize("NFD", str(s).lower().strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s)


def _norm_name(s: str) -> str:
    """Minuscules, sans accents, alphanumérique uniquement."""
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", " ", s).strip()


def _find_col(cols: list[str], patterns: list[str]) -> Optional[str]:
    """Retourne le premier nom de colonne dont la forme normalisée contient un pattern."""
    norm_map = {_norm_col(c): c for c in cols}
    for pat in patterns:
        np = _norm_col(pat)
        for nc, orig in norm_map.items():
            if np == nc or np in nc or nc in np:
                return orig
    return None


def _safe_str(val) -> Optional[str]:
    """Convertit une valeur pandas en str, None si vide/NaN."""
    if val is None:
        return None
    if hasattr(val, "__class__") and val.__class__.__name__ in ("NaT",):
        return None
    try:
        import numpy as np
        if isinstance(val, float) and (val != val):   # NaN check
            return None
    except ImportError:
        pass
    s = str(val).strip()
    return None if s.lower() in {"nan", "none", "nat", "n/a", ""} else s


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class GestmaxCandidate:
    candidate_name: str
    application_date: Optional[str] = None   # "dd/mm/yyyy" ou valeur brute
    cv_filename_hint: Optional[str] = None   # nom de fichier CV si présent dans l'export
    offer_ref: Optional[str] = None
    status: Optional[str] = None
    raw: dict = field(default_factory=dict)  # toutes les colonnes brutes


@dataclass
class GestmaxParseResult:
    candidates: list[GestmaxCandidate]
    detected_columns: dict[str, str]         # rôle → nom de colonne réel
    all_columns: list[str]                   # toutes les colonnes du fichier
    warnings: list[str]


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_gestmax_export(file_bytes: bytes, filename: str) -> GestmaxParseResult:
    """
    Parse un export CSV ou XLSX Gestmax.

    Détecte automatiquement les colonnes candidates courantes.
    Tolère les encodages latin-1 (fréquents dans les exports français).
    """
    warnings: list[str] = []
    ext = Path(filename).suffix.lower()

    # Chargement
    try:
        if ext in {".xlsx", ".xls"}:
            df = pd.read_excel(io.BytesIO(file_bytes), dtype=str)
        elif ext == ".csv":
            # Essai séquencé d'encodages/séparateurs courants
            loaded = False
            for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
                try:
                    df = pd.read_csv(
                        io.BytesIO(file_bytes),
                        sep=None, engine="python",
                        encoding=enc, dtype=str,
                    )
                    loaded = True
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            if not loaded:
                return GestmaxParseResult([], {}, [], ["Impossible de décoder le fichier CSV."])
        else:
            return GestmaxParseResult([], {}, [], [f"Format non supporté : '{ext}'. Utilisez .csv ou .xlsx."])
    except Exception as e:
        return GestmaxParseResult([], {}, [], [f"Erreur de lecture : {e}"])

    if df.empty:
        return GestmaxParseResult([], {}, [], ["Le fichier est vide."])

    # Nettoyage colonnes
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    # Détection automatique des colonnes
    full_col = _find_col(cols, _FULLNAME_PATTERNS)
    last_col = _find_col(cols, _LASTNAME_PATTERNS) if not full_col else None
    first_col = _find_col(cols, _FIRSTNAME_PATTERNS) if not full_col else None
    date_col = _find_col(cols, _DATE_PATTERNS)
    cv_col = _find_col(cols, _CVFILE_PATTERNS)
    offer_col = _find_col(cols, _OFFER_PATTERNS)
    status_col = _find_col(cols, _STATUS_PATTERNS)

    detected: dict[str, str] = {}
    if full_col:
        detected["Nom complet"] = full_col
    if last_col:
        detected["Nom"] = last_col
    if first_col:
        detected["Prénom"] = first_col
    if date_col:
        detected["Date de candidature"] = date_col
    if cv_col:
        detected["Fichier CV"] = cv_col
    if offer_col:
        detected["Offre / Poste"] = offer_col
    if status_col:
        detected["Statut"] = status_col

    has_name = bool(full_col or last_col or first_col)
    if not has_name:
        warnings.append(
            "Colonne 'nom candidat' non détectée automatiquement. "
            f"La première colonne ({cols[0]!r}) sera utilisée comme nom. "
            f"Colonnes disponibles : {', '.join(cols[:10])}"
        )

    # Construction des candidats
    candidates: list[GestmaxCandidate] = []

    for _, row in df.iterrows():
        raw = {k: (_safe_str(v) or "") for k, v in row.items()}

        # Nom
        if full_col:
            name = _safe_str(row.get(full_col)) or ""
        elif last_col or first_col:
            parts = [
                _safe_str(row.get(first_col)) or "" if first_col else "",
                _safe_str(row.get(last_col)) or "" if last_col else "",
            ]
            name = " ".join(p for p in parts if p).strip()
        else:
            name = _safe_str(row.iloc[0]) or ""

        name = name.strip()
        if not name or name.lower() in {"nan", "none", ""}:
            continue

        # Date de candidature
        date_val = None
        if date_col:
            v = _safe_str(row.get(date_col))
            if v:
                # Reformate si c'est une date ISO
                m = re.match(r"(\d{4})-(\d{2})-(\d{2})", v)
                date_val = f"{m.group(3)}/{m.group(2)}/{m.group(1)}" if m else v

        candidates.append(GestmaxCandidate(
            candidate_name=name,
            application_date=date_val,
            cv_filename_hint=_safe_str(row.get(cv_col)) if cv_col else None,
            offer_ref=_safe_str(row.get(offer_col)) if offer_col else None,
            status=_safe_str(row.get(status_col)) if status_col else None,
            raw=raw,
        ))

    if not candidates:
        warnings.append("Aucun candidat extrait — vérifiez le format du fichier.")

    return GestmaxParseResult(
        candidates=candidates,
        detected_columns=detected,
        all_columns=cols,
        warnings=warnings,
    )


# ── Matching CV → candidat ────────────────────────────────────────────────────

def match_cv_to_candidate(
    cv_filename: str,
    candidates: list[GestmaxCandidate],
) -> Optional[GestmaxCandidate]:
    """
    Tente de relier un nom de fichier CV à une entrée Gestmax.

    Stratégies (dans l'ordre, la première qui réussit l'emporte) :
    1. Correspondance exacte sur cv_filename_hint de l'export
    2. Tous les tokens du nom sont présents dans le nom de fichier
    3. Majorité des tokens (≥ 50%, minimum 2) présents dans le nom de fichier

    Retourne None si aucune correspondance convaincante n'est trouvée.
    """
    stem = _norm_name(Path(cv_filename).stem)

    for candidate in candidates:
        # 1. Indice de fichier explicite dans l'export
        if candidate.cv_filename_hint:
            hint_stem = _norm_name(Path(candidate.cv_filename_hint).stem)
            if hint_stem and (hint_stem in stem or stem in hint_stem):
                return candidate

        # Tokens du nom (filtre les particules d'une lettre)
        name_tokens = [t for t in _norm_name(candidate.candidate_name).split() if len(t) > 1]
        if not name_tokens:
            continue

        hit_count = sum(1 for t in name_tokens if t in stem)

        # 2. Correspondance totale
        if hit_count == len(name_tokens):
            return candidate

        # 3. Correspondance majoritaire (prénom + nom suffisent)
        if len(name_tokens) >= 2 and hit_count >= max(2, (len(name_tokens) + 1) // 2):
            return candidate

    return None
