#!/usr/bin/env python3
"""Python MobileCLIP2-S2 retrieval+speed benchmark (counterpart to the Unity Sentis run).

Embeds with the SAME text_model.onnx that Unity Sentis loads (onnxruntime here), using open_clip's
CLIP BPE tokenizer for MobileCLIP2-S2 (context length 77). CLIP-appropriate small chunks so the text
encoder actually represents each passage. Generates the shared query set (and dumps it for Unity).

Usage: python3 bench_python_mobileclip.py <manual> <onnx> [chunkTokens=64] [overlap=16] [numQueries=1000] [topK=10] [queriesOut]
"""
import sys, time, re, random
import numpy as np
import onnxruntime as ort
import open_clip
import tiktoken

manual, onnx_path = sys.argv[1], sys.argv[2]
chunk_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 64
overlap = int(sys.argv[4]) if len(sys.argv) > 4 else 16
num_queries = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
topK = int(sys.argv[6]) if len(sys.argv) > 6 else 10
queries_out = sys.argv[7] if len(sys.argv) > 7 else "/tmp/clip_queries.txt"

enc = tiktoken.encoding_for_model("gpt-4o-mini")
def chunk_text(content, size, ov):
    toks = enc.encode(content); step = size - ov; out = []
    for start in range(0, max(len(toks), 1), step):
        if start >= len(toks): break
        out.append(enc.decode(toks[start:min(start + size, len(toks))]).strip())
    return out

def norm(s): return re.sub(r"\s+", " ", s).strip()

text = open(manual, encoding="utf-8").read()
total_tokens = len(enc.encode(text))
print("================ Python MobileCLIP2-S2 Retrieval Benchmark ================")
print(f"manual    : {manual}")
print(f"corpus    : {len(text.splitlines()):,} lines, {total_tokens:,} tokens")
print(f"onnx      : {onnx_path}")
contents = chunk_text(text, chunk_tokens, overlap)
print(f"chunks    : {len(contents):,} (size={chunk_tokens}, overlap={overlap}) -- CLIP-appropriate")

clip_tok = open_clip.get_tokenizer("MobileCLIP2-S2")
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
print(f"embed dim : {sess.get_outputs()[0].shape}")

def embed(texts):
    ids = clip_tok(list(texts)).numpy().astype(np.int64)  # exported model bakes batch=1
    return np.array([sess.run(None, {in_name: ids[i:i+1]})[0][0] for i in range(ids.shape[0])], dtype=np.float32)

# Phase 1: ingest (embed chunks)
print("\n[Phase 1] Vector-only ingest (chunk + MobileCLIP embed)...")
t0 = time.perf_counter(); mat = embed(contents); ingest = time.perf_counter() - t0
nrm = np.linalg.norm(mat, axis=1, keepdims=True); nrm[nrm == 0] = 1.0; matn = mat / nrm
print(f"  ingest time      : {ingest:.1f}s ({len(contents)/ingest:.1f} chunks/s, {total_tokens/ingest:,.0f} tokens/s)")

# Build query set from the chunks (sentences -> their source chunk). Dump for Unity to reuse.
rng = random.Random(42)
ss = re.compile(r"(?<=[.!?])\s+")
cands = []
for i, c in enumerate(contents):
    for raw in ss.split(c):
        q = norm(raw)
        if 40 <= len(q) <= 160 and re.search("[A-Za-z]{4,}", q):
            cands.append(q)
cands = list(dict.fromkeys(cands)); rng.shuffle(cands)
queries = cands[:num_queries]
open(queries_out, "w", encoding="utf-8").write("\n".join(queries))
print(f"\n[Phase 2] Retrieval: {len(queries):,} queries (dumped to {queries_out})...")

# Whitespace-normalize chunk content for a robust, implementation-independent gold match
# (queries are already normalized, so this makes gold identical across Python and Unity).
ncontents = [norm(c) for c in contents]
def gold_for(q): return {i for i, c in enumerate(ncontents) if q in c}

_ = embed(["warmup"])
hit1 = hit5 = hitK = 0; mrr = 0.0; lat = []
t0 = time.perf_counter()
for qi, q in enumerate(queries):
    gold = gold_for(q)
    s = time.perf_counter()
    qv = embed([q])[0]; qn = qv / (np.linalg.norm(qv) or 1.0)
    sims = matn @ qn
    top = np.argpartition(-sims, min(topK, len(sims)-1))[:topK]; top = top[np.argsort(-sims[top])]
    lat.append((time.perf_counter() - s) * 1000)
    rank = next((r for r, idx in enumerate(top) if idx in gold), -1)
    if rank >= 0:
        if rank == 0: hit1 += 1
        if rank < 5: hit5 += 1
        hitK += 1; mrr += 1.0/(rank+1)
    if (qi+1) % 200 == 0: print(f"  ...{qi+1}/{len(queries)} done")
search = time.perf_counter() - t0

lat.sort(); n = len(queries); pct = lambda p: lat[min(len(lat)-1, int(p*len(lat)))]
print(f"\n---------------- Retrieval quality (n={n}) ----------------")
print(f"  Recall@1   : {hit1*100/n:.1f}%  ({hit1}/{n})")
print(f"  Recall@5   : {hit5*100/n:.1f}%  ({hit5}/{n})")
print(f"  Recall@{topK:<2} : {hitK*100/n:.1f}%  ({hitK}/{n})")
print(f"  MRR@{topK}    : {mrr/n:.3f}")
print(f"\n---------------- Search speed ----------------")
print(f"  total search time: {search:.1f}s for {n} queries")
print(f"  per query  avg   : {np.mean(lat):.1f} ms  (p50 {pct(0.50):.1f} / p95 {pct(0.95):.1f})")
print(f"  throughput       : {n/search:.1f} queries/s")
print("\n================ Benchmark complete ================")
