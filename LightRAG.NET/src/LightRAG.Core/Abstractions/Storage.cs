namespace LightRAG.Core.Abstractions;

/// <summary>
/// A loosely-typed storage record. Mirrors Python's <c>dict[str, Any]</c> so the engine
/// can read/write arbitrary fields without a rigid schema per namespace.
/// </summary>
public sealed class StorageRecord : Dictionary<string, object?>
{
    public StorageRecord() { }

    public StorageRecord(IDictionary<string, object?> source) : base(source) { }
}

/// <summary>Shared storage lifecycle contract, ported from <c>StorageNameSpace</c> in <c>lightrag/base.py</c>.</summary>
public interface IStorageNameSpace
{
    /// <summary>Logical namespace (e.g. "full_docs", "entities").</summary>
    string Namespace { get; }

    /// <summary>Optional workspace for data isolation; empty string when unused.</summary>
    string Workspace { get; }

    /// <summary>Initialize the storage (load from disk, open connections).</summary>
    Task InitializeAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    /// <summary>Finalize the storage (flush, close connections).</summary>
    Task FinalizeAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    /// <summary>Commit buffered operations after indexing (flush to durable storage).</summary>
    Task IndexDoneCallbackAsync(CancellationToken cancellationToken = default);

    /// <summary>Discard any not-yet-flushed buffered index ops (used when a batch aborts).</summary>
    Task DropPendingIndexOpsAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;

    /// <summary>Drop all data and reset to initial state. Returns a status/message pair.</summary>
    Task<(string Status, string Message)> DropAsync(CancellationToken cancellationToken = default);
}

/// <summary>Key-value storage, ported from <c>BaseKVStorage</c>.</summary>
public interface IKvStorage : IStorageNameSpace
{
    Task<StorageRecord?> GetByIdAsync(string id, CancellationToken cancellationToken = default);

    Task<IReadOnlyList<StorageRecord?>> GetByIdsAsync(IReadOnlyList<string> ids, CancellationToken cancellationToken = default);

    /// <summary>Return the subset of <paramref name="keys"/> that do NOT yet exist.</summary>
    Task<ISet<string>> FilterKeysAsync(ISet<string> keys, CancellationToken cancellationToken = default);

    Task UpsertAsync(IReadOnlyDictionary<string, StorageRecord> data, CancellationToken cancellationToken = default);

    Task DeleteAsync(IReadOnlyList<string> ids, CancellationToken cancellationToken = default);

    Task<bool> IsEmptyAsync(CancellationToken cancellationToken = default);
}

/// <summary>Vector similarity storage, ported from <c>BaseVectorStorage</c>.</summary>
public interface IVectorStorage : IStorageNameSpace
{
    EmbeddingFunc EmbeddingFunc { get; }

    float CosineBetterThanThreshold { get; }

    /// <summary>
    /// Query the vector store. When <paramref name="queryEmbedding"/> is provided, it is used
    /// directly; otherwise the query string is embedded. Returns the top-k records, each
    /// including its id, a "distance" score, and any stored meta fields.
    /// </summary>
    Task<IReadOnlyList<StorageRecord>> QueryAsync(
        string query,
        int topK,
        float[]? queryEmbedding = null,
        CancellationToken cancellationToken = default);

    Task UpsertAsync(IReadOnlyDictionary<string, StorageRecord> data, CancellationToken cancellationToken = default);

    Task DeleteAsync(IReadOnlyList<string> ids, CancellationToken cancellationToken = default);

    Task DeleteEntityAsync(string entityName, CancellationToken cancellationToken = default);

    Task DeleteEntityRelationAsync(string entityName, CancellationToken cancellationToken = default);

    Task<StorageRecord?> GetByIdAsync(string id, CancellationToken cancellationToken = default);

    Task<IReadOnlyList<StorageRecord>> GetByIdsAsync(IReadOnlyList<string> ids, CancellationToken cancellationToken = default);

    Task<IReadOnlyDictionary<string, float[]>> GetVectorsByIdsAsync(IReadOnlyList<string> ids, CancellationToken cancellationToken = default);
}

/// <summary>Knowledge graph storage. All edges are undirected. Ported from <c>BaseGraphStorage</c>.</summary>
public interface IGraphStorage : IStorageNameSpace
{
    Task<bool> HasNodeAsync(string nodeId, CancellationToken cancellationToken = default);

    Task<bool> HasEdgeAsync(string sourceNodeId, string targetNodeId, CancellationToken cancellationToken = default);

    Task<int> NodeDegreeAsync(string nodeId, CancellationToken cancellationToken = default);

    Task<int> EdgeDegreeAsync(string srcId, string tgtId, CancellationToken cancellationToken = default);

    Task<StorageRecord?> GetNodeAsync(string nodeId, CancellationToken cancellationToken = default);

    Task<StorageRecord?> GetEdgeAsync(string sourceNodeId, string targetNodeId, CancellationToken cancellationToken = default);

    /// <summary>Get all edges connected to a node as (source, target) pairs, or null if the node is absent.</summary>
    Task<IReadOnlyList<(string Source, string Target)>?> GetNodeEdgesAsync(string sourceNodeId, CancellationToken cancellationToken = default);

    Task UpsertNodeAsync(string nodeId, IReadOnlyDictionary<string, object?> nodeData, CancellationToken cancellationToken = default);

    Task UpsertEdgeAsync(string sourceNodeId, string targetNodeId, IReadOnlyDictionary<string, object?> edgeData, CancellationToken cancellationToken = default);

    Task DeleteNodeAsync(string nodeId, CancellationToken cancellationToken = default);

    Task RemoveNodesAsync(IReadOnlyList<string> nodes, CancellationToken cancellationToken = default);

    Task RemoveEdgesAsync(IReadOnlyList<(string Source, string Target)> edges, CancellationToken cancellationToken = default);

    /// <summary>Get all node labels (entity names), sorted alphabetically.</summary>
    Task<IReadOnlyList<string>> GetAllLabelsAsync(CancellationToken cancellationToken = default);

    Task<IReadOnlyList<StorageRecord>> GetAllNodesAsync(CancellationToken cancellationToken = default);

    Task<IReadOnlyList<StorageRecord>> GetAllEdgesAsync(CancellationToken cancellationToken = default);

    Task<KnowledgeGraph> GetKnowledgeGraphAsync(string nodeLabel, int maxDepth = 3, int maxNodes = 1000, CancellationToken cancellationToken = default);

    // ---- Batch helpers with serial default implementations (override for native batch) ----

    async Task<IReadOnlyDictionary<string, StorageRecord>> GetNodesBatchAsync(IReadOnlyList<string> nodeIds, CancellationToken cancellationToken = default)
    {
        var result = new Dictionary<string, StorageRecord>();
        foreach (var nodeId in nodeIds)
        {
            var node = await GetNodeAsync(nodeId, cancellationToken).ConfigureAwait(false);
            if (node is not null)
            {
                result[nodeId] = node;
            }
        }
        return result;
    }

    async Task<IReadOnlyDictionary<string, int>> NodeDegreesBatchAsync(IReadOnlyList<string> nodeIds, CancellationToken cancellationToken = default)
    {
        var result = new Dictionary<string, int>();
        foreach (var nodeId in nodeIds)
        {
            result[nodeId] = await NodeDegreeAsync(nodeId, cancellationToken).ConfigureAwait(false);
        }
        return result;
    }

    async Task<IReadOnlyDictionary<string, IReadOnlyList<(string Source, string Target)>>> GetNodesEdgesBatchAsync(
        IReadOnlyList<string> nodeIds, CancellationToken cancellationToken = default)
    {
        var result = new Dictionary<string, IReadOnlyList<(string, string)>>();
        foreach (var nodeId in nodeIds)
        {
            var edges = await GetNodeEdgesAsync(nodeId, cancellationToken).ConfigureAwait(false);
            result[nodeId] = edges ?? [];
        }
        return result;
    }
}

/// <summary>Document-status storage, ported from <c>DocStatusStorage</c>.</summary>
public interface IDocStatusStorage : IKvStorage
{
    Task<IReadOnlyDictionary<string, int>> GetStatusCountsAsync(CancellationToken cancellationToken = default);

    Task<IReadOnlyDictionary<string, DocProcessingStatus>> GetDocsByStatusAsync(DocStatus status, CancellationToken cancellationToken = default);

    Task<IReadOnlyDictionary<string, DocProcessingStatus>> GetDocsByStatusesAsync(IReadOnlyList<DocStatus> statuses, CancellationToken cancellationToken = default);
}
