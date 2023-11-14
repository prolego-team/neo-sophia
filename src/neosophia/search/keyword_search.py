"""BM25-based keyword search."""

import numpy as np

from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer

from rank_bm25 import BM25Okapi

from neosophia.search.utils.data_utils import SearchResult

try:
    STOPWORDS = stopwords.words('english')
except:
    import nltk
    nltk.download('stopwords')
    STOPWORDS = stopwords.words('english')

try:
    word_tokenize('This is a test')
except:
    import nltk
    nltk.download('punkt')

try:
    WordNetLemmatizer().lemmatize('testing')
except:
    import nltk
    nltk.download('wordnet')


LEMMATIZE = WordNetLemmatizer().lemmatize


def tokenize(texts: list[str]) -> list[list[str]]:
    """Tokenize texts with lemmatization."""
    tokens = [
        [
            LEMMATIZE(token)
            for token in word_tokenize(text.lower())
            if token not in STOPWORDS
        ]
        for text in texts
    ]
    return tokens


def build_index(corpus: list[str]) -> BM25Okapi:
    """Tokenize and index a corpus for BM25."""
    return BM25Okapi(tokenize(corpus))


def bm25_topk(query: str, index: BM25Okapi, top_k: int):
    """Get the top-k BM25 matches from an index."""
    scores = index.get_scores(tokenize([query,])[0])
    top_k_inds = list(np.argsort(scores, )[-top_k:])[::-1]
    return top_k_inds, list(scores[top_k_inds])


def keyword_search(
        index: BM25Okapi,
        query: str,
        ids: list[tuple[int]],
        chunks: list[str],
        top_k: int
    ) -> list[SearchResult]:
    """Get the top-k BM25 matches as a list of SearchResults."""
    indices, scores = bm25_topk(query, index, top_k)

    results = []
    for chunk_id,score in zip(indices, scores):
        file = ids[chunk_id][0]
        tree_ind = ids[chunk_id][1:-1]
        paragraph = ids[chunk_id][-1]
        result = {
            'similarity_score': score,
            'file': file,
            'tree_index': tree_ind,
            'paragraph_index': paragraph,
            'chunk_id': int(chunk_id),
            'text': chunks[chunk_id],
            'reranked_score': -1000
        }
        results.append(SearchResult(**result))

    return results
