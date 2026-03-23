# Dockerfile pour le projet PDF Coherence Checker (offline POC)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    STREAMLIT_SERVER_PORT=8501

# créer un utilisateur non-root
RUN useradd -m appuser || true

WORKDIR /app

# Copier uniquement requirements d'abord pour profiter du cache docker
COPY requirements.txt /app/

# Installer dépendances système nécessaires (minimales) et dépendances Python
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       curl \
       libglib2.0-0 \
       ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet (inclut le modèle local dans ./models si présent)
COPY . /app

# Assurer que l'utilisateur non-root possède les fichiers
RUN chown -R appuser:appuser /app

USER appuser

ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.headless", "true", "--server.enableCORS", "false"]
