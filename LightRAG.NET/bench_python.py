#!/usr/bin/env python3
"""Python LightRAG retrieval+speed benchmark — 1:1 counterpart to LightRAG.Benchmark (C#).

Uses LightRAG's real chunker (`chunking_by_token_size` + `TiktokenTokenizer`) and the same Ollama
embedding model, then evaluates the SAME query set (shared file) with the same content-based gold
metric (a hit = a retrieved chunk whose text contains the query sentence). Cosine search via numpy
(mathematically identical to the nano-vectordb store LightRAG ships).

Usage: python3 bench_python.py <manual> <queries_file> [topK=10] [embed_model=nomic-embed-text] [dim=768]
"""
import sys, time, re
import numpy as np
import ollama
from lightrag.utils import TiktokenTokenizer
from lightrag.chunker.token_size import chunking_by_token_size

manual = sys.argv[1]
queries_file = sys.argv[2]
topK = int(sys.argv[3]) if len(sys.argv) > 3 else 10
embed_model = sys.argv[4] if len(sys.argv) > 4 else "nomic-embed-text"
host = "http://localhost:11434"
batch = 10

text = open(manual, encoding="utf-8").read()
tok = TiktokenTokenizer("gpt-4o-mini")
total_tokens = len(tok.encode(text))

print("================ Python LightRAG Retrieval Benchmark ================")
print(f"manual    : {manual}")
print(f"corpus    : {len(text.splitlines()):,} lines, {len(text):,} chars, {total_tokens:,} tokens")
print(f"models    : embed={embed_model} @ {host}")

# Authentic LightRAG chunking.
chunks = chunking_by_token_size(tok, text, chunk_overlap_token_size=100, chunk_token_size=1200)
contents = [c["content"] for c in chunks]
print(f"chunks    : {len(chunks):,} (size=1200, overlap=100)")

client = ollama.Client(host=host)

def embed(texts):
    out = []
    for i in range(0, len(texts), batch):
        resp = client.embed(model=embed_model, input=texts[i:i+batch])
        out.extend(resp["embeddings"])
    return np.array(out, dtype=np.float32)

# ---------- Phase 1: vector-only ingest (embed chunks) ----------
print("\n[Phase 1] Vector-only ingest (chunk + embed)...")
t0 = time.perf_counter()
mat = embed(contents)
ingest = time.perf_counter() - t0
norms = np.linalg.norm(mat, axis=1, keepdims=True)
norms[norms == 0] = 1.0
matn = mat / norms
print(f"  ingest time      : {ingest:.1f}s")
print(f"  throughput       : {len(chunks)/ingest:.1f} chunks/s, {total_tokens/ingest:,.0f} tokens/s")

# ---------- Phase 2: retrieval over the shared query set ----------
queries = [l for l in open(queries_file, encoding="utf-8").read().splitlines() if l.strip()]
print(f"\n[Phase 2] Retrieval: {len(queries):,} queries (loaded from {queries_file})...")

# Precompute gold sets (chunk indices whose content contains the query) — content-based, like C#.
def gold_for(q):
    return {i for i, c in enumerate(contents) if q in c}

# warm up
client.embed(model=embed_model, input=["warmup"])

hit1 = hit5 = hitK = 0
mrr = 0.0
lat = []
t0 = time.perf_counter()
for qi, q in enumerate(queries):
    gold = gold_for(q)
    s = time.perf_counter()
    qv = np.array(client.embed(model=embed_model, input=[q])["embeddings"][0], dtype=np.float32)
    qn = qv / (np.linalg.norm(qv) or 1.0)
    sims = matn @ qn
    top = np.argpartition(-sims, min(topK, len(sims)-1))[:topK]
    top = top[np.argsort(-sims[top])]
    lat.append((time.perf_counter() - s) * 1000)
    rank = next((r for r, idx in enumerate(top) if idx in gold), -1)
    if rank >= 0:
        if rank == 0: hit1 += 1
        if rank < 5: hit5 += 1
        hitK += 1
        mrr += 1.0 / (rank + 1)
    if (qi + 1) % 200 == 0:
        print(f"  ...{qi+1}/{len(queries)} done")
search = time.perf_counter() - t0

lat.sort()
n = len(queries)
pct = lambda p: lat[min(len(lat)-1, int(p*len(lat)))]
print(f"\n---------------- Retrieval quality (n={n}) ----------------")
print(f"  Recall@1   : {hit1*100/n:.1f}%  ({hit1}/{n})")
print(f"  Recall@5   : {hit5*100/n:.1f}%  ({hit5}/{n})")
print(f"  Recall@{topK:<2} : {hitK*100/n:.1f}%  ({hitK}/{n})")
print(f"  MRR@{topK}    : {mrr/n:.3f}")
print(f"\n---------------- Search speed ----------------")
print(f"  total search time: {search:.1f}s for {n} queries")
print(f"  per query  avg   : {np.mean(lat):.1f} ms")
print(f"  per query  p50   : {pct(0.50):.1f} ms")
print(f"  per query  p95   : {pct(0.95):.1f} ms")
print(f"  per query  min/max: {lat[0]:.1f} / {lat[-1]:.1f} ms")
print(f"  throughput       : {n/search:.1f} queries/s")
print("\n================ Benchmark complete ================")
