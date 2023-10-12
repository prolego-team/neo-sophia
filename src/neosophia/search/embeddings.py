from typing import Any
import pickle

import torch

from sentence_transformers import SentenceTransformer, util, CrossEncoder

Model = Any


def get_model(model: str) -> Model:
    """Get an embedding model."""
    return SentenceTransformer(model)


def encode(texts: list[str], model: Model):
    """Get embeddings for texts."""
    show_progress = len(texts)>10
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=show_progress)


def query(target_embeddings: list[torch.Tensor], query_embedding: torch.Tensor, model: Model, top_k: int = 5):
    """Query a set of embeddings"""
    # query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, target_embeddings, top_k=top_k)
    return hits


def load_embeddings(fname: str):
    with open(fname, 'rb') as f:
        embeddings = pickle.load(f)['embeddings']

    return embeddings


def save_embeddings(embeddings, fname: str):
    with open(fname, 'wb') as fout:
        pickle.dump({'embeddings': embeddings}, fout, protocol=pickle.HIGHEST_PROTOCOL)


class ReRanker:

    def __init__(self, model: str):
        self.model = CrossEncoder(model)

    def cross_encode(self, query: str, context: str) -> float:
        score = self.model.predict([(query, context)])[0]
        return score