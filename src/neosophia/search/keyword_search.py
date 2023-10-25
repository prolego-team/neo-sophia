import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer

from rank_bm25 import BM25Okapi

from neosophia.search.utils.data_utils import SearchResult

STOPWORDS = stopwords.words('english')
word_tokenizer = word_tokenize
lemmatizer = WordNetLemmatizer().lemmatize

def tokenize(texts: list[str]) -> list[list[str]]:
    tokens = [
        [
            lemmatizer(token)
            for token in word_tokenize(text.lower())
            if token not in STOPWORDS
        ]
        for text in texts
    ]
    return tokens
    # return [text.lower().split(' ') for text in texts]


def build_index(corpus: list[str]) -> BM25Okapi:
    return BM25Okapi(tokenize(corpus))


def bm25_topk(query: str, index: BM25Okapi, top_k: int):
    scores = index.get_scores(tokenize([query,])[0])
    top_k_inds = list(np.argsort(scores, )[-top_k:])[::-1]
    return top_k_inds, list(scores[top_k_inds])


def keyword_search(index: BM25Okapi, query: str, ids: list[tuple[int]], chunks: list[str], top_k: int):
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
