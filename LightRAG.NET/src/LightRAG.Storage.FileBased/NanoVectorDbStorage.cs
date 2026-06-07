using LightRAG.Core.Abstractions;
using LightRAG.Core.Configuration;
using LightRAG.Core.Utils;
using Newtonsoft.Json.Linq;

namespace LightRAG.Storage.FileBased;

/// <summary>
/// In-memory cosine-similarity vector store, a clean reimplementation of
/// <c>NanoVectorDBStorage</c> (<c>lightrag/kg/nano_vector_db_impl.py</c> + the <c>nano-vectordb</c> lib).
/// Embedding is deferred to flush time so repeated upserts of the same id coalesce into one
/// embedding pass. Vectors are persisted as plain JSON float arrays (clean format — no
/// float16/zlib/base64 byte-compat with the Python files).
/// </summary>
public sealed class NanoVectorDbStorage : FileStorageBase, IVectorStorage
{
    private sealed class Entry
    {
        public required float[] Vector { get; set; }
        public required StorageRecord Data { get; init; } // includes content, meta fields, created_at
    }

    private readonly Dictionary<string, Entry> _data = new();
    private readonly Dictionary<string, StorageRecord> _pending = new();
    private readonly string _filePath;
    private readonly int _batchNum;
    private readonly HashSet<string>? _metaFields;
    private bool _dirty;

    public EmbeddingFunc EmbeddingFunc { get; }

    public float CosineBetterThanThreshold { get; }

    public NanoVectorDbStorage(
        string workingDir,
        string @namespace,
        EmbeddingFunc embeddingFunc,
        float cosineBetterThanThreshold = Constants.DefaultCosineThreshold,
        string workspace = "",
        int embeddingBatchNum = Constants.DefaultEmbeddingBatchNum,
        IReadOnlyCollection<string>? metaFields = null)
        : base(workingDir, @namespace, workspace)
    {
        EmbeddingFunc = embeddingFunc;
        CosineBetterThanThreshold = cosineBetterThanThreshold;
        _batchNum = embeddingBatchNum;
        // When set, only these caller fields are persisted (plus id/created_at), matching Python's
        // meta_fields whitelist. When null, all caller fields are kept (back-compat for direct use).
        _metaFields = metaFields is null ? null : [.. metaFields];
        _filePath = Path.Combine(Directory, $"vdb_{@namespace}.json");
    }

    public Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        EnsureDirectory();
        SweepOrphanTmp(_filePath);
        LoadFromDisk();
        return Task.CompletedTask;
    }

    private void LoadFromDisk()
    {
        _data.Clear();
        _pending.Clear();
        if (!File.Exists(_filePath))
        {
            return;
        }
        var root = JObject.Parse(File.ReadAllText(_filePath));
        if (root["data"] is not JArray dataArray)
        {
            return;
        }
        foreach (var item in dataArray)
        {
            if (item is not JObject jo)
            {
                continue;
            }
            var record = JsonHelper.ToRecord(jo);
            var id = record.GetString("id");
            if (id is null || !record.TryGetValue("vector", out var vectorObj) || vectorObj is not List<object?> vectorList)
            {
                continue;
            }
            var vector = vectorList.Select(Convert.ToSingle).ToArray();
            record.Remove("vector");
            _data[id] = new Entry { Vector = vector, Data = record };
        }
    }

    public Task UpsertAsync(IReadOnlyDictionary<string, StorageRecord> data, CancellationToken cancellationToken = default)
    {
        if (data.Count == 0)
        {
            return Task.CompletedTask;
        }
        var createdAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        foreach (var (id, record) in data)
        {
            StorageRecord stored;
            if (_metaFields is null)
            {
                stored = new StorageRecord(record) { ["id"] = id, ["created_at"] = createdAt };
            }
            else
            {
                // Persist only whitelisted meta fields (plus id/created_at), like Python's nano-vectordb.
                stored = new StorageRecord { ["id"] = id, ["created_at"] = createdAt };
                foreach (var (k, v) in record)
                {
                    if (_metaFields.Contains(k))
                    {
                        stored[k] = v;
                    }
                }
            }
            _pending[id] = stored; // overwrites prior pending for same id (re-embed)
        }
        return Task.CompletedTask;
    }

    public async Task<IReadOnlyList<StorageRecord>> QueryAsync(
        string query,
        int topK,
        float[]? queryEmbedding = null,
        CancellationToken cancellationToken = default)
    {
        float[] embedding;
        if (queryEmbedding is not null)
        {
            embedding = queryEmbedding;
        }
        else
        {
            var embedded = await EmbeddingFunc.InvokeAsync([query], context: "query", cancellationToken).ConfigureAwait(false);
            embedding = embedded[0];
        }

        var queryNorm = VectorMath.Norm(embedding);
        var scored = new List<(double Score, string Id, Entry Entry)>(_data.Count);
        foreach (var (id, entry) in _data)
        {
            var score = VectorMath.CosineSimilarity(embedding, entry.Vector, queryNorm);
            if (score >= CosineBetterThanThreshold)
            {
                scored.Add((score, id, entry));
            }
        }

        scored.Sort((a, b) => b.Score.CompareTo(a.Score));

        var results = new List<StorageRecord>(Math.Min(topK, scored.Count));
        foreach (var (score, id, entry) in scored.Take(topK))
        {
            var record = new StorageRecord(entry.Data) { ["id"] = id, ["distance"] = score };
            results.Add(record);
        }
        return results;
    }

    public Task DeleteAsync(IReadOnlyList<string> ids, CancellationToken cancellationToken = default)
    {
        foreach (var id in ids)
        {
            if (_data.Remove(id) | _pending.Remove(id))
            {
                _dirty = true;
            }
        }
        return Task.CompletedTask;
    }

    public Task DeleteEntityAsync(string entityName, CancellationToken cancellationToken = default)
    {
        var id = Hashing.ComputeMdHashId(entityName, "ent-");
        return DeleteAsync([id], cancellationToken);
    }

    public Task DeleteEntityRelationAsync(string entityName, CancellationToken cancellationToken = default)
    {
        // Remove relationship vectors whose source or target is this entity.
        var toRemove = _data
            .Where(kv => kv.Value.Data.GetString("src_id") == entityName || kv.Value.Data.GetString("tgt_id") == entityName)
            .Select(kv => kv.Key)
            .ToList();
        foreach (var id in _pending
            .Where(kv => kv.Value.GetString("src_id") == entityName || kv.Value.GetString("tgt_id") == entityName)
            .Select(kv => kv.Key).ToList())
        {
            _pending.Remove(id);
            _dirty = true;
        }
        return DeleteAsync(toRemove, cancellationToken);
    }

    public Task<StorageRecord?> GetByIdAsync(string id, CancellationToken cancellationToken = default)
    {
        if (_data.TryGetValue(id, out var entry))
        {
            return Task.FromResult<StorageRecord?>(new StorageRecord(entry.Data) { ["id"] = id });
        }
        if (_pending.TryGetValue(id, out var pending))
        {
            return Task.FromResult<StorageRecord?>(new StorageRecord(pending));
        }
        return Task.FromResult<StorageRecord?>(null);
    }

    public async Task<IReadOnlyList<StorageRecord>> GetByIdsAsync(IReadOnlyList<string> ids, CancellationToken cancellationToken = default)
    {
        var result = new List<StorageRecord>();
        foreach (var id in ids)
        {
            var record = await GetByIdAsync(id, cancellationToken).ConfigureAwait(false);
            if (record is not null)
            {
                result.Add(record);
            }
        }
        return result;
    }

    public Task<IReadOnlyDictionary<string, float[]>> GetVectorsByIdsAsync(IReadOnlyList<string> ids, CancellationToken cancellationToken = default)
    {
        var result = new Dictionary<string, float[]>();
        foreach (var id in ids)
        {
            if (_data.TryGetValue(id, out var entry))
            {
                result[id] = entry.Vector;
            }
        }
        return Task.FromResult<IReadOnlyDictionary<string, float[]>>(result);
    }

    public async Task IndexDoneCallbackAsync(CancellationToken cancellationToken = default)
    {
        await FlushPendingAsync(cancellationToken).ConfigureAwait(false);
        if (_dirty)
        {
            Persist();
            _dirty = false;
        }
    }

    private async Task FlushPendingAsync(CancellationToken cancellationToken)
    {
        if (_pending.Count == 0)
        {
            return;
        }

        var items = _pending.ToList();
        var contents = items.Select(kv => kv.Value.GetString("content") ?? string.Empty).ToList();

        var vectors = new List<float[]>(contents.Count);
        for (var i = 0; i < contents.Count; i += _batchNum)
        {
            var batch = contents.Skip(i).Take(_batchNum).ToList();
            var embedded = await EmbeddingFunc.InvokeAsync(batch, context: "document", cancellationToken).ConfigureAwait(false);
            vectors.AddRange(embedded);
        }

        if (vectors.Count != items.Count)
        {
            throw new InvalidOperationException(
                $"Embedding count {vectors.Count} does not match pending count {items.Count}.");
        }

        for (var i = 0; i < items.Count; i++)
        {
            _data[items[i].Key] = new Entry { Vector = vectors[i], Data = items[i].Value };
        }
        _pending.Clear();
        _dirty = true;
    }

    public Task FinalizeAsync(CancellationToken cancellationToken = default) => IndexDoneCallbackAsync(cancellationToken);

    public Task DropPendingIndexOpsAsync(CancellationToken cancellationToken = default)
    {
        _pending.Clear();
        return Task.CompletedTask;
    }

    public Task<(string Status, string Message)> DropAsync(CancellationToken cancellationToken = default)
    {
        _data.Clear();
        _pending.Clear();
        _dirty = false;
        try
        {
            if (File.Exists(_filePath))
            {
                File.Delete(_filePath);
            }
            return Task.FromResult(("success", "data dropped"));
        }
        catch (Exception ex)
        {
            return Task.FromResult(("error", ex.Message));
        }
    }

    private void Persist()
    {
        EnsureDirectory();
        var payload = new Dictionary<string, object?>
        {
            ["embedding_dim"] = EmbeddingFunc.EmbeddingDim,
            ["data"] = _data.Select(kv =>
            {
                var record = new StorageRecord(kv.Value.Data) { ["id"] = kv.Key, ["vector"] = kv.Value.Vector };
                return record;
            }).ToList(),
        };
        AtomicWrite(_filePath, JsonHelper.Serialize(payload));
    }
}
