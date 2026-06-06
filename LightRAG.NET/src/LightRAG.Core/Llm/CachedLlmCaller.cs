using LightRAG.Core.Abstractions;
using LightRAG.Core.Concurrency;
using LightRAG.Core.Utils;

namespace LightRAG.Core.Llm;

/// <summary>
/// Concrete <see cref="ILlmCaller"/> that routes a role's calls through the priority scheduler and an
/// optional KV-backed response cache. Combines the responsibilities of <c>use_llm_func_with_cache</c>,
/// the per-role priority wrapping (<c>llm_roles.py</c>), and <c>handle_cache</c>/<c>save_to_cache</c>.
/// </summary>
public sealed class CachedLlmCaller : ILlmCaller
{
    private readonly ILlmModel _model;
    private readonly PriorityAsyncScheduler _scheduler;
    private readonly IKvStorage? _cache;
    private readonly int _priority;
    private readonly double? _temperature;

    public CachedLlmCaller(
        ILlmModel model,
        PriorityAsyncScheduler scheduler,
        IKvStorage? cache = null,
        int priority = Configuration.Constants.DefaultProcessingPriority,
        double? temperature = null)
    {
        _model = model;
        _scheduler = scheduler;
        _cache = cache;
        _priority = priority;
        _temperature = temperature;
    }

    public async Task<string> CompleteAsync(
        string userPrompt,
        string? systemPrompt = null,
        IReadOnlyList<ChatMessage>? history = null,
        bool jsonMode = false,
        string cacheType = "default",
        string? chunkId = null,
        CancellationToken cancellationToken = default)
    {
        var historyKey = history is null ? string.Empty
            : string.Concat(history.Select(m => $"{m.Role}:{m.Content}\n"));
        var hash = Hashing.ComputeArgsHash(cacheType, systemPrompt ?? string.Empty, userPrompt, historyKey, jsonMode ? "1" : "0");

        if (_cache is not null)
        {
            var cached = await _cache.GetByIdAsync(hash, cancellationToken).ConfigureAwait(false);
            if (cached is not null && cached.GetString("return") is { } cachedReturn)
            {
                return cachedReturn;
            }
        }

        var options = new LlmCompletionOptions
        {
            SystemPrompt = systemPrompt,
            History = history ?? [],
            JsonMode = jsonMode,
            Temperature = _temperature,
        };

        var result = await _scheduler.RunAsync(
            ct => _model.CompleteAsync(userPrompt, options, ct),
            _priority,
            cancellationToken: cancellationToken).ConfigureAwait(false);

        if (_cache is not null && !string.IsNullOrEmpty(result))
        {
            await _cache.UpsertAsync(new Dictionary<string, StorageRecord>
            {
                [hash] = new() { ["return"] = result, ["cache_type"] = cacheType },
            }, cancellationToken).ConfigureAwait(false);
        }

        return result;
    }
}
