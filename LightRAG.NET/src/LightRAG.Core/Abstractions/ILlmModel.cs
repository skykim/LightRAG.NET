namespace LightRAG.Core.Abstractions;

/// <summary>A single chat message in a conversation history.</summary>
/// <param name="Role">"user", "assistant", or "system".</param>
/// <param name="Content">Message text.</param>
public sealed record ChatMessage(string Role, string Content);

/// <summary>
/// Options for an LLM completion call. Folds the Python free-function kwargs
/// (<c>system_prompt</c>, <c>history_messages</c>, <c>response_format</c>, ...) into a typed object.
/// </summary>
public sealed record LlmCompletionOptions
{
    /// <summary>Optional system prompt prepended to the conversation.</summary>
    public string? SystemPrompt { get; init; }

    /// <summary>Prior conversation turns (used for the gleaning loop).</summary>
    public IReadOnlyList<ChatMessage> History { get; init; } = [];

    /// <summary>When true, request a JSON object response (maps to Ollama <c>format=json</c>).</summary>
    public bool JsonMode { get; init; }

    /// <summary>Sampling temperature; null uses the provider/model default.</summary>
    public double? Temperature { get; init; }
}

/// <summary>
/// Abstraction over an LLM completion provider, ported from the free
/// <c>*_model_complete</c> functions in <c>lightrag/llm/*</c>.
/// </summary>
public interface ILlmModel
{
    /// <summary>The underlying model name (e.g. "qwen2.5:latest").</summary>
    string ModelName { get; }

    /// <summary>Complete a prompt and return the full response text.</summary>
    Task<string> CompleteAsync(
        string prompt,
        LlmCompletionOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>Complete a prompt, streaming the response as it is produced.</summary>
    IAsyncEnumerable<string> StreamAsync(
        string prompt,
        LlmCompletionOptions? options = null,
        CancellationToken cancellationToken = default);
}
