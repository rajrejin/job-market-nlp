"""Exact and near-duplicate detection (normalised text hash and MinHash LSH)."""

from __future__ import annotations

import hashlib
import re

from datasketch import MinHash, MinHashLSH

TOKEN = re.compile(r"\w+", re.UNICODE)


def normalised_hash(text: str) -> str:
    toks = TOKEN.findall(text.lower())
    return hashlib.sha1(" ".join(toks).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    toks = TOKEN.findall(text.lower())
    if len(toks) < k:
        return {" ".join(toks)}
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=0)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster_near_duplicates(texts: list[str], threshold: float = 0.9) -> list[int]:
    """Cluster id per text (the smallest index in its cluster)."""
    uf = UnionFind(len(texts))
    by_hash: dict[str, int] = {}
    for i, t in enumerate(texts):
        h = normalised_hash(t)
        if h in by_hash:
            uf.union(by_hash[h], i)
        else:
            by_hash[h] = i
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for i, t in enumerate(texts):
        m = minhash(t)
        for j in lsh.query(m):
            uf.union(int(j), i)
        lsh.insert(str(i), m)
    return [uf.find(i) for i in range(len(texts))]
