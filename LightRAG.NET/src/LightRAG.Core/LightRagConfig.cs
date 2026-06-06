using LightRAG.Core.Abstractions;
using LightRAG.Core.Configuration;
using LightRAG.Core.Extraction;
using LightRAG.Core.Pipeline;
using LightRAG.Core.Query;

namespace LightRAG.Core;

/// <summary>Top-level configuration for a <see cref="LightRag"/> instance.</summary>
public sealed record LightRagConfig
{
    public string Language { get; init; } = Constants.DefaultSummaryLanguage;

    /// <summary>Max concurrent LLM calls (the priority scheduler bound).</summary>
    public int MaxAsync { get; init; } = Constants.DefaultMaxAsync;

    /// <summary>Per-call LLM timeout. Null disables the timeout.</summary>
    public TimeSpan? LlmTimeout { get; init; } = TimeSpan.FromSeconds(Constants.DefaultLlmTimeout);

    /// <summary>LLM sampling temperature; null uses the provider default.</summary>
    public double? Temperature { get; init; }

    /// <summary>Whether to cache LLM responses in the llm_response_cache KV store.</summary>
    public bool EnableLlmCache { get; init; } = true;

    public PipelineOptions Pipeline { get; init; } = new();
    public ExtractionOptions Extraction { get; init; } = new();
    public MergeOptions Merge { get; init; } = new();
    public QueryOptions Query { get; init; } = new();
}

/// <summary>The storage backends a <see cref="LightRag"/> instance operates over (all interface-typed).</summary>
public sealed record LightRagStorages
{
    public required IKvStorage FullDocs { get; init; }
    public required IKvStorage TextChunks { get; init; }
    public required IKvStorage LlmResponseCache { get; init; }
    public required IDocStatusStorage DocStatus { get; init; }
    public required IVectorStorage EntitiesVdb { get; init; }
    public required IVectorStorage RelationshipsVdb { get; init; }
    public required IVectorStorage ChunksVdb { get; init; }
    public required IGraphStorage Graph { get; init; }

    /// <summary>All storages, for lifecycle (initialize/flush/finalize) iteration.</summary>
    public IReadOnlyList<IStorageNameSpace> All =>
    [
        FullDocs, TextChunks, LlmResponseCache, DocStatus,
        EntitiesVdb, RelationshipsVdb, ChunksVdb, Graph,
    ];
}
