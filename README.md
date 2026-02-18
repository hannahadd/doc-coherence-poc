# PDF Coherence Checker (Offline POC)

**Objectif (manager demo)** : comparer deux documents (ex. *CV* vs *offre d'emploi*) en local (**sans cloud**) et produire :
- un **score de cohérence**,
- la liste des **compétences manquantes / matchées**,
- des **preuves** (extraits de texte) pour justifier le résultat,
- un **rapport JSON** téléchargeable.

> Ce POC est volontairement **simple, explicable et 100% offline**. Il sert de base à la transposition vers le cas
> *documentation technique* vs *support de formation*.

---

## 1) Quickstart

### Prérequis
- Python 3.10+ (fonctionne aussi en 3.9)
- Installation des dépendances :

```bash
pip install -r requirements.txt
```

### Lancer l'app
```bash
streamlit run streamlit_app.py
```

---

## 2) Démo incluse (sans PDF)
L'onglet **"Demo dataset"** permet de sélectionner un couple *(resume, job description)* depuis un jeu de données
embarqué dans `data/huggingface_resume_job_fit_RAW.xlsx`.

---

## 3) Mode PDF
Onglet **"Compare PDFs"** :
- Upload CV (PDF)
- Upload offre (PDF)

⚠️ Si les PDFs sont **scannés**, l'extraction de texte peut être vide (OCR non inclus dans ce POC).
On pourra ajouter une brique OCR (Tesseract + pdf2image) en étape 2.

---

## 4) Comment le score est calculé (explicable)
Le score final (0–100) combine :
- **Skill coverage** : proportion de compétences de l'offre retrouvées dans le CV (avec synonymes).
- **Text similarity** : similarité TF‑IDF sur **caractères** (robuste FR/EN, peu sensible aux stopwords).

Formule (par défaut) :
- `score = 0.65 * coverage + 0.35 * similarity`

Les pondérations sont dans `src/scoring.py`.

---

## 5) Roadmap vers le cas "documentation vs formation"
- Ajout OCR (si PDF scannés)
- Extraction structurée : sections, tableaux, numéros de versions, exigences, prérequis
- Détection de contradictions : valeurs numériques, versions, “must/shall”, procédures
- Indexation + evidence retrieval (RAG **local**) pour justifier chaque incohérence
- Hardening SSI : allow-list formats, logs, redaction, mode air‑gapped, etc.

---

## 6) Structure du projet
```
doc_coherence_poc/
  streamlit_app.py
  requirements.txt
  src/
    pdf_utils.py
    skills_reference.py
    comparator.py
    scoring.py
  data/
    skills_reference.json
    jobs_dataset.json
    huggingface_resume_job_fit_RAW.xlsx
```

---

## 7) Notes SSI / confidentialité
- Aucune requête réseau n'est effectuée.
- Aucun appel à un service cloud / LLM externe.
- Les fichiers uploadés sont traités en mémoire et ne sont pas envoyés ailleurs.
