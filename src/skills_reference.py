import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, List, Tuple


_TOKEN_RE = re.compile(r"[a-zA-Z0-9\+\#\.\-]+")


def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def ngrams(tokens: List[str], n: int) -> Set[str]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass
class SkillReference:
    canon2variants: Dict[str, Set[str]]
    max_len: int

    @staticmethod
    def load(path: str) -> "SkillReference":
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        # attendu dans ton fichier:
        # - technical_skills: list[str]
        # - variations: dict[str, list[str]]
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

        return SkillReference(canon2variants=canon2variants, max_len=max_len)

    def extract(self, text: str) -> Set[str]:
        tokens = tokenize(text)

        grams_by_n = {}
        for n in range(1, self.max_len + 1):
            grams_by_n[n] = ngrams(tokens, n)

        found = set()
        for canon, variants in self.canon2variants.items():
            for v in variants:
                n = len(v.split())
                if n <= 0:
                    continue
                if v in grams_by_n.get(n, set()):
                    found.add(canon)
                    break
        return found
