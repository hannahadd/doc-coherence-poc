from sentence_transformers import SentenceTransformer


def load_embedding_model(model_path_or_name: str) -> SentenceTransformer:
    """
    model_path_or_name:
      - chemin local: "models/all-MiniLM-L6-v2"
      - ou nom HF si tu es sur une machine connectée (à éviter en défense)
    """
    return SentenceTransformer(model_path_or_name)
