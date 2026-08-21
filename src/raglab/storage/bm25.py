"""自实现 BM25：不依赖检索库，便于在面试中讲清楚每个公式。"""

from __future__ import annotations

import math
import re
from collections import Counter

_LATIN_RE = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """中英混合分词：
    - 英文按单词；
    - 中文用字符二元组（bigram），兼顾效率和召回。
    """
    text = text.lower()
    tokens = _LATIN_RE.findall(text)
    for cjk in _CJK_RE.findall(text):
        if len(cjk) == 1:
            tokens.append(cjk)
        else:
            tokens.extend(cjk[i : i + 2] for i in range(len(cjk) - 1))
    return tokens


class BM25Index:
    """BM25 倒排索引（k1、b 可调，默认取常见经验值）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._texts: dict[str, str] = {}
        self._term_freqs: dict[str, Counter[str]] = {}
        self._doc_freq: Counter[str] = Counter()
        self._doc_len: dict[str, int] = {}
        self._avgdl = 0.0

    @property
    def size(self) -> int:
        return len(self._texts)

    def add(self, chunk_id: str, text: str) -> None:
        tokens = tokenize(text)
        self._texts[chunk_id] = text
        self._term_freqs[chunk_id] = Counter(tokens)
        self._doc_len[chunk_id] = len(tokens)
        for term in set(tokens):
            self._doc_freq[term] += 1
        self._avgdl = sum(self._doc_len.values()) / len(self._doc_len)

    def _idf(self, df: int) -> float:
        n = self.size
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        if not query_terms or not self._texts:
            return []
        scores: dict[str, float] = {}
        for term in set(query_terms):
            df = self._doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = self._idf(df)
            for chunk_id, tf_counter in self._term_freqs.items():
                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_len[chunk_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
