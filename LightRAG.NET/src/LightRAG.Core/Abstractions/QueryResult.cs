namespace LightRAG.Core.Abstractions;

/// <summary>
/// Unified query result for all query modes, ported from <c>QueryResult</c> in <c>lightrag/base.py</c>.
/// </summary>
public sealed class QueryResult
{
    /// <summary>Text content for non-streaming responses.</summary>
    public string? Content { get; init; }

    /// <summary>Streaming response iterator for streaming responses.</summary>
    public IAsyncEnumerable<string>? ResponseIterator { get; init; }

    /// <summary>Complete structured data (references, metadata, retrieved context).</summary>
    public IReadOnlyDictionary<string, object?>? RawData { get; init; }

    /// <summary>Whether this is a streaming result.</summary>
    public bool IsStreaming { get; init; }
}

/// <summary>
/// Query context result, ported from <c>QueryContextResult</c> in <c>lightrag/base.py</c>.
/// </summary>
/// <param name="Context">LLM context string.</param>
/// <param name="RawData">Complete structured data, including the reference list.</param>
public sealed record QueryContextResult(string Context, IReadOnlyDictionary<string, object?> RawData);
