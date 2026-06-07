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
    private readonly string _role;

    public CachedLlmCaller(
        ILlmModel model,
        PriorityAsyncScheduler scheduler,
        IKvStorage? cache = null,
        int priority = Configuration.Constants.DefaultProcessingPriority,
        double? temperature = null,
        string role = "default")
    {
        _model = model;
        _scheduler = scheduler;
        _cache = cache;
        _priority = priority;
        _temperature = temperature;
        _role = role;
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
        var cacheKey = BuildCacheKey(userPrompt, systemPrompt, history, jsonMode, cacheType);

        if (_cache is not null)
        {
            var cached = await _cache.GetByIdAsync(cacheKey, cancellationToken).ConfigureAwait(false);
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
            // Duplicate-write suppression (matches save_to_cache): skip if identical content cached.
            var existing = await _cache.GetByIdAsync(cacheKey, cancellationToken).ConfigureAwait(false);
            if (existing is null || existing.GetString("return") != result)
            {
                await _cache.UpsertAsync(new Dictionary<string, StorageRecord>
                {
                    [cacheKey] = new()
                    {
                        ["return"] = result,
                        ["cache_type"] = cacheType,
                        ["chunk_id"] = chunkId,
                        ["original_prompt"] = userPrompt,
                    },
                }, cancellationToken).ConfigureAwait(false);
            }
        }

        return result;
    }

    /// <summary>
    /// Build the flattened cache key <c>default:{cache_type}:{md5}</c>, where the md5 hashes the
    /// prompt plus the response-format and non-secret LLM identity (role + model). Ports
    /// <c>use_llm_func_with_cache</c> + <c>generate_cache_key</c> so a model swap invalidates the cache.
    /// </summary>
    private string BuildCacheKey(string userPrompt, string? systemPrompt, IReadOnlyList<ChatMessage>? history, bool jsonMode, string cacheType)
    {
        // _prompt = "\n".join([user, system, history]) over non-empty parts; history is JSON of the messages.
        var parts = new List<string>();
        if (!string.IsNullOrEmpty(userPrompt)) { parts.Add(userPrompt); }
        if (!string.IsNullOrEmpty(systemPrompt)) { parts.Add(systemPrompt!); }
        if (history is { Count: > 0 })
        {
            var msgs = history.Select(m => JsonHelper.SerializeLine(("role", m.Role), ("content", m.Content)));
            parts.Add("[" + string.Join(", ", msgs) + "]");
        }
        var prompt = string.Join("\n", parts);

        // _serialize_cache_variant: compact JSON (",":"," separators), sorted keys, "" for None.
        var responseFormatKey = jsonMode ? "{\"type\":\"json_object\"}" : string.Empty;
        var modelJson = JsonHelper.Serialize(_model.ModelName, indented: false);
        var roleJson = JsonHelper.Serialize(_role, indented: false);
        var llmIdentityKey = $"{{\"binding\":null,\"host\":null,\"model\":{modelJson},\"role\":{roleJson}}}";

        var argHash = Hashing.ComputeArgsHash(
            prompt, "\n<response_format>\n", responseFormatKey, "\n<llm_identity>\n", llmIdentityKey);
        return $"default:{cacheType}:{argHash}";
    }
}
