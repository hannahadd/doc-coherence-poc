# ─────────────────────────────────────────────────────────────
#  doc_coherence_poc  –  Streamlit app (offline, CPU-only)
#
#  AVANT DE BUILDER : remplace les deux placeholders ci-dessous
#
#  1) HARBOR_REGISTRY  → ex: harbor.mon-entreprise.fr
#  2) HARBOR_PROJECT   → ex: proxy-dockerhub  (le projet Harbor
#                         qui proxifie Docker Hub, à demander à
#                         ton collègue ou à l'IT)
# ─────────────────────────────────────────────────────────────
FROM HARBOR_REGISTRY/HARBOR_PROJECT/python:3.13-slim

# ── Métadonnées ──────────────────────────────────────────────
LABEL maintainer="ton.email@entreprise.fr" \
      app="doc-coherence-poc" \
      version="1.0"

# ── Variables d'environnement ─────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    PYTHONPATH=/app \
    # Mode offline : aucun appel réseau HuggingFace au démarrage
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    # Désactive les télémétries
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # Évite le conflit libomp entre torch et faiss (surtout macOS, inoffensif sur Linux)
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    # Tesseract : pointe sur les fichiers de langue embarqués dans l'image
    # (évite la dépendance aux packs apt tesseract-ocr-fra / tesseract-ocr-eng)
    TESSDATA_PREFIX=/app/models/tessdata

# ── Dépendances système ───────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       # requis par torch / numpy
       libgomp1 \
       # requis par faiss-cpu
       libglib2.0-0 \
       # utile pour healthcheck et debug
       curl \
       # OCR images (multimodal) — binaire uniquement
       # Les fichiers de langue (fra/eng) sont embarqués dans models/tessdata/
       # => pas besoin de tesseract-ocr-fra / tesseract-ocr-eng depuis apt
       # => fonctionne même sans accès aux dépôts Debian externes
       tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# ── Création d'un utilisateur non-root ────────────────────────
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# ── Dépendances Python ────────────────────────────────────────
# On copie requirements.txt en premier pour profiter du cache Docker :
# si requirements.txt ne change pas, cette couche n'est pas reconstruite.
COPY requirements.txt /app/

# PYPI_MIRROR → URL du miroir PyPI interne (Artifactory / Nexus / autre)
#               ex: https://artifactory.mon-entreprise.fr/artifactory/api/pypi/pypi-virtual/simple
#               À demander à ton collègue ou à l'IT.
#
# Note sur torch : le miroir interne distribue probablement la version
# standard (avec CUDA, ~2,5 Go). Si le miroir a la variante cpu-only, utilise :
#   torch==2.3.1+cpu
# Sinon laisse torch==2.3.1 — ça fonctionne en CPU, c'est juste plus lourd.
ARG PYPI_MIRROR=https://PYPI_MIRROR_URL/simple

RUN pip install --upgrade pip --no-cache-dir \
       --index-url ${PYPI_MIRROR} \
    && pip install --no-cache-dir \
       --index-url ${PYPI_MIRROR} \
       -r requirements.txt

# ── Code source + données + modèle embarqué ───────────────────
# Le .dockerignore exclut .venv, __pycache__, .git, etc.
COPY --chown=appuser:appuser . /app

# ── Passage en utilisateur non-root ──────────────────────────
USER appuser

# ── Port exposé ───────────────────────────────────────────────
EXPOSE 8501

# ── Health check (utilisé par Docker / Rancher / K8s) ─────────
# Vérifie que Streamlit répond toutes les 30s, échec après 3 tentatives
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --silent --fail http://localhost:8501/_stcore/health || exit 1

# ── Démarrage ─────────────────────────────────────────────────
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port", "8501", \
     "--server.headless", "true", \
     "--server.enableCORS", "false", \
     "--server.enableXsrfProtection", "false"]
