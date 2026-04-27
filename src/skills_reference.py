import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, List


_TOKEN_RE = re.compile(r"[a-zA-Z0-9\+\#\.\-]+")
_ALPHA_ONLY_RE = re.compile(r"^[a-z0-9]+$")

# Compétences courtes connues : bypass du filtre min_chars.
# Ce sont des vrais noms de langages/outils qui s'écrivent en 1-2 lettres.
# On les accepte même courts car ils apparaissent souvent seuls dans une liste de compétences.
# Contrepartie connue : "c" peut matcher "c'est" (le ' sépare), "r" peut matcher "R&D".
# C'est un compromis acceptable vs. les rater complètement.
_SHORT_SKILL_WHITELIST: Set[str] = {
    # "r" et "c" retirés : génèrent des faux positifs massifs sur textes français
    # (ex. "rédiger" → token "r", "c'est" → token "c")
    "go",                    # langage
    "ai", "ml", "dl",        # domaines IA
    "bi", "etl",             # data
    "qa", "ui", "ux",        # métiers
}

MIN_ALPHA_CHARS = 3


def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(text: str) -> List[str]:
    """Tokenise le texte. Les points internes sont conservés (node.js, .net)
    mais les points de fin de token (ponctuation de fin de phrase) sont retirés."""
    raw = _TOKEN_RE.findall((text or "").lower())
    return [t.strip(".") for t in raw if t.strip(".")]


def ngrams(tokens: List[str], n: int) -> Set[str]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _is_matchable(variant: str) -> bool:
    """
    Rejette les variants trop courts et ambigus.
    - Whitelist explicite (r, c, go…) → toujours ok, ce sont de vrais noms de langages
    - Caractère spécial (c++, c#, .net, node.js) → toujours ok
    - Sinon : doit avoir au moins MIN_ALPHA_CHARS caractères
    """
    if variant in _SHORT_SKILL_WHITELIST:
        return True
    if not _ALPHA_ONLY_RE.match(variant):
        return True  # contient +, #, ., -, / → distinctif
    return len(variant) >= MIN_ALPHA_CHARS


@dataclass
class SkillReference:
    canon2variants: Dict[str, Set[str]]
    max_len: int

    @staticmethod
    def _load_one(path: str) -> tuple[Dict[str, Set[str]], int]:
        """Charge un seul fichier JSON et retourne (canon2variants, max_len)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        technical = set(normalize(x) for x in data.get("technical_skills", []) if isinstance(x, str))
        variations = data.get("variations", {}) or {}

        canon2variants: Dict[str, Set[str]] = {}
        max_len = 1

        for canon, vars_ in variations.items():
            canon_n = normalize(canon)
            varset = {canon_n}
            if isinstance(vars_, list):
                for v in vars_:
                    if isinstance(v, str) and v.strip():
                        varset.add(normalize(v))
            canon2variants[canon_n] = varset
            for v in varset:
                max_len = max(max_len, len(v.split()))

        for t in technical:
            canon2variants.setdefault(t, {t})
            max_len = max(max_len, len(t.split()))

        return canon2variants, max_len

    @staticmethod
    def load(path: str) -> "SkillReference":
        """Charge un seul fichier de compétences."""
        canon2variants, max_len = SkillReference._load_one(path)
        return SkillReference(canon2variants=canon2variants, max_len=max_len)

    @staticmethod
    def load_multi(paths: List[str]) -> "SkillReference":
        """
        Fusionne plusieurs fichiers de compétences (base générale + fichier entreprise).
        Les fichiers suivants enrichissent le premier sans écraser ses entrées.
        """
        merged: Dict[str, Set[str]] = {}
        max_len = 1

        for path in paths:
            if not path or not Path(path).exists():
                continue
            c2v, ml = SkillReference._load_one(path)
            for canon, variants in c2v.items():
                if canon in merged:
                    merged[canon] |= variants
                else:
                    merged[canon] = set(variants)
            max_len = max(max_len, ml)

        return SkillReference(canon2variants=merged, max_len=max_len)

    def extract(self, text: str) -> Set[str]:
        tokens = tokenize(text)

        grams_by_n = {}
        for n in range(1, self.max_len + 1):
            grams_by_n[n] = ngrams(tokens, n)

        found = set()
        for canon, variants in self.canon2variants.items():
            for v in variants:
                if not _is_matchable(v):
                    continue  # trop court/ambigu → on skip ce variant
                n = len(v.split())
                if n <= 0:
                    continue
                if v in grams_by_n.get(n, set()):
                    found.add(canon)
                    break
        return found
