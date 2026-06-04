"""Local retrieval helpers for BrowseComp-Plus evaluation."""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass


@dataclass
class RetrievedDoc:
    docid: str
    text: str
    score: float | None = None


class BM25Retriever:
    def __init__(self, index_path: str):
        from searcher.searchers.bm25_searcher import BM25Searcher

        ns = argparse.Namespace(index_path=index_path)
        self.searcher = BM25Searcher(ns)
        # Pyserini/Lucene search can throw JVM NPE under concurrent calls.
        # Serialize BM25 access for stability.
        self._lock = threading.Lock()

    def search(self, query: str, k: int) -> list[RetrievedDoc]:
        with self._lock:
            hits = self.searcher.search(query, k=k)
        return [
            RetrievedDoc(
                docid=str(x.get("docid", "")),
                text=str(x.get("text", "")),
                score=float(x["score"]) if x.get("score") is not None else None,
            )
            for x in hits
        ]
