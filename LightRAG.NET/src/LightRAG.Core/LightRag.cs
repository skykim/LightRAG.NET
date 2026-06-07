using LightRAG.Core.Abstractions;
using LightRAG.Core.Concurrency;
using LightRAG.Core.Configuration;
using LightRAG.Core.Extraction;
using LightRAG.Core.Llm;
using LightRAG.Core.Pipeline;
using LightRAG.Core.Query;

namespace LightRAG.Core;

/// <summary>
/// Top-level LightRAG facade, ported from the orchestration role of the <c>LightRAG</c> dataclass in
/// <c>lightrag/lightrag.py</c>. Wires storages, role-bound LLM callers (via the priority scheduler and
/// cache), the ingestion pipeline, and the query engine. The Python <c>global_config</c> dict is
/// replaced by typed configuration and constructor injection.
/// </summary>
public sealed class LightRag
{
    private readonly LightRagStorages _storages;
    private readonly DocumentPipeline _pipeline;
    private readonly QueryEngine _queryEngine;

    public LightRag(
        LightRagConfig config,
        ILlmModel llm,
        ITokenizer tokenizer,
        LightRagStorages storages)
    {
        _storages = storages;

        var scheduler = new PriorityAsyncScheduler(config.MaxAsync, config.LlmTimeout);
        var cache = config.EnableLlmCache ? storages.LlmResponseCache : null;

        // Role-bound callers differ only by scheduling priority so interactive queries
        // preempt background ingestion work in the shared priority queue.
        // The role partitions the LLM cache identity (extraction & summary share the "extract" role,
        // matching Python's get_llm_cache_identity usage).
        var extractCaller = new CachedLlmCaller(llm, scheduler, cache, Constants.DefaultProcessingPriority, config.Temperature, role: "extract");
        var summaryCaller = new CachedLlmCaller(llm, scheduler, cache, Constants.DefaultSummaryPriority, config.Temperature, role: "extract");
        var queryCaller = new CachedLlmCaller(llm, scheduler, cache, Constants.DefaultQueryPriority, config.Temperature, role: "query");

        var extractor = new EntityExtractor(extractCaller, tokenizer, config.Extraction);
        var builder = new KnowledgeGraphBuilder(
            storages.Graph, storages.EntitiesVdb, storages.RelationshipsVdb, tokenizer, summaryCaller, config.Merge);

        _pipeline = new DocumentPipeline(
            tokenizer, extractor, builder,
            storages.FullDocs, storages.TextChunks, storages.ChunksVdb, storages.DocStatus,
            storages.All, config.Pipeline);

        _queryEngine = new QueryEngine(
            storages.Graph, storages.EntitiesVdb, storages.RelationshipsVdb, storages.ChunksVdb,
            storages.TextChunks, tokenizer, queryCaller, queryCaller, config.Query);
    }

    /// <summary>Initialize all storages (load from disk / open connections).</summary>
    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        foreach (var storage in _storages.All)
        {
            await storage.InitializeAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>Ingest a single text document.</summary>
    public Task<int> InsertAsync(string content, string filePath = "unknown_source", CancellationToken cancellationToken = default)
        => _pipeline.InsertAsync([new InputDocument(content, filePath)], cancellationToken);

    /// <summary>Ingest multiple documents.</summary>
    public Task<int> InsertAsync(IReadOnlyList<InputDocument> documents, CancellationToken cancellationToken = default)
        => _pipeline.InsertAsync(documents, cancellationToken);

    /// <summary>Query the knowledge base and generate an answer.</summary>
    public Task<QueryResult> QueryAsync(string query, QueryParam? param = null, string? systemPrompt = null, CancellationToken cancellationToken = default)
        => _queryEngine.QueryAsync(query, param, systemPrompt, cancellationToken);

    /// <summary>
    /// Retrieve the top-k chunk records for a query directly from the chunk vector store, without
    /// LLM answer generation. Each record includes its chunk "id", "content", and a "distance" score.
    /// Useful for retrieval benchmarking.
    /// </summary>
    public Task<IReadOnlyList<StorageRecord>> RetrieveChunksAsync(string query, int topK, CancellationToken cancellationToken = default)
        => _storages.ChunksVdb.QueryAsync(query, topK, cancellationToken: cancellationToken);

    /// <summary>Flush and finalize all storages.</summary>
    public async Task FinalizeAsync(CancellationToken cancellationToken = default)
    {
        foreach (var storage in _storages.All)
        {
            await storage.FinalizeAsync(cancellationToken).ConfigureAwait(false);
        }
    }
}
