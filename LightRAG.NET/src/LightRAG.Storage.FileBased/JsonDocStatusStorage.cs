using LightRAG.Core.Abstractions;
using LightRAG.Core.Utils;
using Newtonsoft.Json.Linq;

namespace LightRAG.Storage.FileBased;

/// <summary>
/// JSON-file document-status store, ported from <c>JsonDocStatusStorage</c>
/// (<c>lightrag/kg/json_doc_status_impl.py</c>). Persists to <c>kv_store_{namespace}.json</c>.
/// </summary>
public sealed class JsonDocStatusStorage : FileStorageBase, IDocStatusStorage
{
    private readonly Dictionary<string, StorageRecord> _data = new();
    private readonly string _filePath;
    private bool _dirty;

    public JsonDocStatusStorage(string workingDir, string @namespace = NameSpace.DocStatus, string workspace = "")
        : base(workingDir, @namespace, workspace)
    {
        _filePath = Path.Combine(Directory, $"kv_store_{@namespace}.json");
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
        if (!File.Exists(_filePath))
        {
            return;
        }
        var root = JObject.Parse(File.ReadAllText(_filePath));
        foreach (var property in root.Properties())
        {
            if (property.Value is JObject jo)
            {
                _data[property.Name] = JsonHelper.ToRecord(jo);
            }
        }
    }

    public Task<StorageRecord?> GetByIdAsync(string id, CancellationToken cancellationToken = default)
        => Task.FromResult(_data.TryGetValue(id, out var record) ? new StorageRecord(record) : null);

    public Task<IReadOnlyList<StorageRecord?>> GetByIdsAsync(IReadOnlyList<string> ids, CancellationToken cancellationToken = default)
    {
        IReadOnlyList<StorageRecord?> result = ids
            .Select(id => _data.TryGetValue(id, out var record) ? new StorageRecord(record) : null)
            .ToList();
        return Task.FromResult(result);
    }

    public Task<ISet<string>> FilterKeysAsync(ISet<string> keys, CancellationToken cancellationToken = default)
    {
        ISet<string> missing = keys.Where(k => !_data.ContainsKey(k)).ToHashSet();
        return Task.FromResult(missing);
    }

    public Task UpsertAsync(IReadOnlyDictionary<string, StorageRecord> data, CancellationToken cancellationToken = default)
    {
        if (data.Count == 0)
        {
            return Task.CompletedTask;
        }
        foreach (var (id, record) in data)
        {
            _data[id] = new StorageRecord(record);
        }
        _dirty = true;
        return Task.CompletedTask;
    }

    public Task DeleteAsync(IReadOnlyList<string> ids, CancellationToken cancellationToken = default)
    {
        foreach (var id in ids)
        {
            if (_data.Remove(id))
            {
                _dirty = true;
            }
        }
        return Task.CompletedTask;
    }

    public Task<bool> IsEmptyAsync(CancellationToken cancellationToken = default) => Task.FromResult(_data.Count == 0);

    public Task<IReadOnlyDictionary<string, int>> GetStatusCountsAsync(CancellationToken cancellationToken = default)
    {
        var counts = new Dictionary<string, int>();
        foreach (var record in _data.Values)
        {
            var status = record.GetString("status") ?? "unknown";
            counts[status] = counts.GetValueOrDefault(status) + 1;
        }
        return Task.FromResult<IReadOnlyDictionary<string, int>>(counts);
    }

    public Task<IReadOnlyDictionary<string, DocProcessingStatus>> GetDocsByStatusAsync(DocStatus status, CancellationToken cancellationToken = default)
        => GetDocsByStatusesAsync([status], cancellationToken);

    public Task<IReadOnlyDictionary<string, DocProcessingStatus>> GetDocsByStatusesAsync(IReadOnlyList<DocStatus> statuses, CancellationToken cancellationToken = default)
    {
        var wanted = statuses.Select(s => s.ToWireValue()).ToHashSet();
        var result = new Dictionary<string, DocProcessingStatus>();
        foreach (var (id, record) in _data)
        {
            var statusValue = record.GetString("status");
            if (statusValue is not null && wanted.Contains(statusValue))
            {
                result[id] = ToDocStatus(record);
            }
        }
        return Task.FromResult<IReadOnlyDictionary<string, DocProcessingStatus>>(result);
    }

    private static DocProcessingStatus ToDocStatus(StorageRecord record)
    {
        var chunksList = record.TryGetValue("chunks_list", out var cl) && cl is List<object?> list
            ? list.Select(x => x?.ToString() ?? string.Empty).ToList()
            : [];
        return new DocProcessingStatus
        {
            ContentSummary = record.GetString("content_summary") ?? string.Empty,
            ContentLength = record.GetInt("content_length"),
            FilePath = record.GetString("file_path") ?? "unknown_source",
            Status = DocStatusExtensions.FromWireValue(record.GetString("status") ?? "pending"),
            CreatedAt = record.GetString("created_at") ?? string.Empty,
            UpdatedAt = record.GetString("updated_at") ?? string.Empty,
            TrackId = record.GetString("track_id"),
            ChunksCount = record.TryGetValue("chunks_count", out var cc) && cc is not null ? record.GetInt("chunks_count") : null,
            ChunksList = chunksList,
            ErrorMsg = record.GetString("error_msg"),
            ContentHash = record.GetString("content_hash"),
        };
    }

    public Task IndexDoneCallbackAsync(CancellationToken cancellationToken = default)
    {
        if (_dirty)
        {
            EnsureDirectory();
            AtomicWrite(_filePath, JsonHelper.Serialize(_data));
            _dirty = false;
        }
        return Task.CompletedTask;
    }

    public Task FinalizeAsync(CancellationToken cancellationToken = default) => IndexDoneCallbackAsync(cancellationToken);

    public Task DropPendingIndexOpsAsync(CancellationToken cancellationToken = default)
    {
        LoadFromDisk();
        _dirty = false;
        return Task.CompletedTask;
    }

    public Task<(string Status, string Message)> DropAsync(CancellationToken cancellationToken = default)
    {
        _data.Clear();
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
}
