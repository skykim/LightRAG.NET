# LightRAG.NET

A C# port of the core of [LightRAG](https://github.com/HKUDS/LightRAG) — knowledge-graph retrieval-augmented
generation — targeting **netstandard2.1** so the same assemblies run on .NET 9 and inside Unity (Mono/IL2CPP).

## Scope

Ported: document ingestion → token chunking → LLM entity/relationship extraction (with gleaning) →
knowledge-graph construction (merge, description summarization) → all query modes (local / global / hybrid /
mix / naive / bypass) → answer generation. File-based storage only (zero external services). Ollama provider.

Out of scope (vs. Python): REST API server, web UI, production DB backends (Postgres/Neo4j/…), non-Ollama
providers, document parsers (PDF/DOCX), and the multi-process shared-storage layer.

## Projects

| Project | TFM | Purpose |
|---|---|---|
| `src/LightRAG.Core` | netstandard2.1 | Engine: abstractions, prompts, tokenizer, scheduler, extraction, KG builder, query engine, pipeline, `LightRag` facade |
| `src/LightRAG.Storage.FileBased` | netstandard2.1 | JSON KV / doc-status, nano-vectordb (cosine), GraphML graph; `FileBasedLightRag` factory |
| `src/LightRAG.Providers.Ollama` | netstandard2.1 | `OllamaLlm` + `OllamaEmbedding` via OllamaSharp |
| `tests/LightRAG.Core.Tests` | net9.0 | 41 unit/integration tests (xUnit) |
| `samples/LightRAG.Sample` | net9.0 | BMW-manual smoke test / demo CLI |

## Quick start (.NET)

```bash
dotnet test                                   # 41 tests, offline (fake LLM + deterministic embeddings)
dotnet run --project samples/LightRAG.Sample  # BMW manual smoke test against local Ollama
```

The sample ingests a slice of `lightrag-unity/Assets/StreamingAssets/Manual/bmw5_manual.txt` with
`gemma4:e2b` + `nomic-embed-text`, then validates retrieval against real sentences from the manual.

## Verification highlights

- **Tokenizer parity**: token counts match Python `tiktoken` (`gpt-4o-mini` / o200k_base) exactly.
- **Chunker parity**: chunk boundaries match the Python `token_size` chunker.
- **End-to-end on real data**: the BMW manual smoke test passes — naive retrieval surfaces real manual
  sentences (e.g. *"Apply the parking brake"*) and hybrid KG queries produce grounded, cited answers.

## Unity

See `../lightrag-unity/Assets/LightRAG/README.md`. Refresh the Unity plugin DLLs with
`./build-unity-plugins.sh`.

## Design notes

- Python's `global_config` dict is replaced by typed configuration + constructor injection.
- The priority-weighted async scheduler (`PriorityAsyncScheduler`) preserves LightRAG's call-ordering
  semantics (interactive queries preempt ingestion) using a `SortedSet` wait queue.
- JSON uses Newtonsoft (works on both .NET and Unity); `json_repair` is reimplemented in `JsonRepair`.
- netstandard2.1 polyfills for `init`/`required` live in `Polyfills.shared.cs`.
