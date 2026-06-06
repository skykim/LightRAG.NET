using LightRAG.Core.Abstractions;

namespace LightRAG.Core.Llm;

/// <summary>
/// A role-bound LLM caller that wraps the underlying provider with the priority scheduler and the
/// response cache. The engine (extraction / query) depends only on this; the orchestrator supplies
/// the concrete implementation. Ported from the <c>use_llm_func_with_cache</c> call sites in
/// <c>operate.py</c>, with the role/priority/cache concerns folded into the implementation.
/// </summary>
public interface ILlmCaller
{
    /// <summary>
    /// Complete a prompt for this role, consulting/populating the cache.
    /// </summary>
    /// <param name="userPrompt">The user prompt.</param>
    /// <param name="systemPrompt">Optional system prompt.</param>
    /// <param name="history">Prior turns (for the gleaning loop).</param>
    /// <param name="jsonMode">Request a JSON object response.</param>
    /// <param name="cacheType">Cache partition tag (e.g. "extract", "keywords", "query").</param>
    /// <param name="chunkId">Optional originating chunk id, for cache locality.</param>
    Task<string> CompleteAsync(
        string userPrompt,
        string? systemPrompt = null,
        IReadOnlyList<ChatMessage>? history = null,
        bool jsonMode = false,
        string cacheType = "default",
        string? chunkId = null,
        CancellationToken cancellationToken = default);
}
