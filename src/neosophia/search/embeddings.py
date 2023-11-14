"""Wrapper for common embedding functions."""

from typing import Any
import pickle

import torch

from sentence_transformers import SentenceTransformer, util

Model = Any


def get_model(model: str) -> Model:
    """Get an embedding model."""
    return SentenceTransformer(model)


def encode(texts: list[str], model: Model):
    """Get embeddings for texts."""
    show_progress = len(texts)>10
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=show_progress)


def query(
        target_embeddings: torch.Tensor,
        query_embedding: torch.Tensor,
        top_k: int = 5
    ):
    """Query a set of embeddings"""
    hits = util.semantic_search(query_embedding, target_embeddings, top_k=top_k)
    return hits


def load_embeddings(fname: str):
    """Load embeddings from file."""
    with open(fname, 'rb') as f:
        embeddings = pickle.load(f)['embeddings']

    return embeddings


def save_embeddings(embeddings, fname: str):
    """Write embeddings to file."""
    with open(fname, 'wb') as fout:
        pickle.dump({'embeddings': embeddings}, fout, protocol=pickle.HIGHEST_PROTOCOL)
