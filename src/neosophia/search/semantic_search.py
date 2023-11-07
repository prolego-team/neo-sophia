"""Functions for similarity and cross-encoder search."""

import torch

import neosophia.search.embeddings as emb
from neosophia.search.utils import doctree
from neosophia.search.utils.data_utils import SearchResult


def cosine_search(
        query_emb: torch.Tensor,
        embeddings: torch.Tensor,
        ids: list[tuple[int]],
        chunks: list[str],
        top_k: int,
    ) -> list[SearchResult]:
    """Use cosine distance to get the `top_k` most similar text chunks to `query`.

    Each embedding (row of the tensor) should coincide with an entry in `ids` and
    `chunks`.  I.e., `embeddings[i,:] ~ ids[i] ~ chunks[i]`.  Each chunk is a
    piece of text taken from index id from the DocTree.
    """
    # Get similarity scores and identifiers
    hits = emb.query(embeddings, query_emb, top_k)[0]

    results = []
    for hit in hits:
        score = hit['score']
        chunk_id = hit['corpus_id']
        file = ids[chunk_id][0]
        ind = ids[chunk_id][1:-1]
        paragraph = ids[chunk_id][-1]

        result = {
            'similarity_score': float(score),
            'file': file,
            'tree_index': ind,
            'paragraph_index': paragraph,
            'chunk_id': chunk_id,
            'text': chunks[chunk_id],
            'reranked_score':-1000
        }
        results.append(SearchResult(**result))

    return results


def rerank(
        inputs: list[SearchResult],
        doc_trees: dict[str, doctree.DocTree],
        query: str,
        rerank_model: emb.Model,
        post_expand: bool
    ):
    """Re-rank a list of results from `cosine_search`.

    `inputs` should be a list of dicts containing at least `text` and `tree_index`.
    """

    # Re-rank
    for result in inputs:
        text = result.text

        if post_expand:
            file = result.file
            tree_ind = result.tree_index
            expanded_items = doctree.expand(text, doc_trees[file], tree_ind)
            max_score = -10000
            for item in expanded_items:
                score = rerank_model.predict([(query, item)])[0]
                if score>max_score:
                    result.reranked_score = float(score)
                    result.text = item
        else:
            result.reranked_score = float(rerank_model.predict([(query, text)])[0])

    # Re sort
    results = sorted(inputs, key=lambda res: res.reranked_score, reverse=True)

    return results
