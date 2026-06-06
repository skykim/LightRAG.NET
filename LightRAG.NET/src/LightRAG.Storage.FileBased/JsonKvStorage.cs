using LightRAG.Core.Abstractions;
using LightRAG.Core.Utils;
using Newtonsoft.Json.Linq;

namespace LightRAG.Storage.FileBased;

/// <summary>
/// JSON-file key-value store, ported from <c>JsonKVStorage</c> in <c>lightrag/kg/json_kv_impl.py</c>.
/// Data lives in memory and is flushed to <c>kv_store_{namespace}.json</c> on
/// <see cref="IndexDoneCallbackAsync"/>. Multi-process sync is out of scope.
/// </summary>
public sealed class JsonKvStorage : FileStorageBase, IKvStorage
{
    private readonly Dictionary<string, StorageRecord> _data = new();
    private readonly string _filePath;
    private bool _dirty;

    public JsonKvStorage(string workingDir, string @namespace, string workspace = "")
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
        => Task.FromResult(_data.TryGetValue(id, out var record) ? Clone(record) : null);

    public Task<IReadOnlyList<StorageRecord?>> GetByIdsAsync(IReadOnlyList<string> ids, CancellationToken cancellationToken = default)
    {
        IReadOnlyList<StorageRecord?> result = ids
            .Select(id => _data.TryGetValue(id, out var record) ? Clone(record) : null)
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
            _data[id] = Clone(record);
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

    public Task<bool> IsEmptyAsync(CancellationToken cancellationToken = default)
        => Task.FromResult(_data.Count == 0);

    public Task IndexDoneCallbackAsync(CancellationToken cancellationToken = default)
    {
        if (_dirty)
        {
            Persist();
            _dirty = false;
        }
        return Task.CompletedTask;
    }

    public Task FinalizeAsync(CancellationToken cancellationToken = default) => IndexDoneCallbackAsync(cancellationToken);

    public Task DropPendingIndexOpsAsync(CancellationToken cancellationToken = default)
    {
        // Re-read committed state, discarding uncommitted in-memory changes.
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

    private void Persist()
    {
        EnsureDirectory();
        AtomicWrite(_filePath, JsonHelper.Serialize(_data));
    }

    private static StorageRecord Clone(StorageRecord record) => new(record);
}
