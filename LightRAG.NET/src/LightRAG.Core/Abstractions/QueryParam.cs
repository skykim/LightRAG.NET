using LightRAG.Core.Configuration;

namespace LightRAG.Core.Abstractions;

/// <summary>Retrieval mode for a query, ported from <c>QueryParam.mode</c>.</summary>
public enum QueryMode
{
    /// <summary>Context-dependent, entity-focused retrieval.</summary>
    Local,

    /// <summary>Global, relationship-focused retrieval.</summary>
    Global,

    /// <summary>Combines local and global retrieval.</summary>
    Hybrid,

    /// <summary>Basic vector search over chunks, no knowledge graph.</summary>
    Naive,

    /// <summary>Knowledge graph (hybrid) plus direct chunk vector retrieval.</summary>
    Mix,

    /// <summary>No retrieval; query sent directly to the LLM.</summary>
    Bypass,
}

/// <summary>
/// Configuration parameters for query execution, ported from <c>QueryParam</c> in <c>lightrag/base.py</c>.
/// </summary>
public sealed record QueryParam
{
    /// <summary>Retrieval mode. Defaults to <see cref="QueryMode.Mix"/>.</summary>
    public QueryMode Mode { get; init; } = QueryMode.Mix;

    /// <summary>If true, only returns the retrieved context without generating a response.</summary>
    public bool OnlyNeedContext { get; init; }

    /// <summary>If true, only returns the generated prompt without producing a response.</summary>
    public bool OnlyNeedPrompt { get; init; }

    /// <summary>Defines the response format, e.g. "Multiple Paragraphs", "Bullet Points".</summary>
    public string ResponseType { get; init; } = "Multiple Paragraphs";

    /// <summary>If true, enables streaming output.</summary>
    public bool Stream { get; init; }

    /// <summary>Number of top items to retrieve (entities in local, relationships in global).</summary>
    public int TopK { get; init; } = Constants.DefaultTopK;

    /// <summary>Number of text chunks to retrieve from vector search and keep after reranking.</summary>
    public int ChunkTopK { get; init; } = Constants.DefaultChunkTopK;

    /// <summary>Max tokens allocated for entity context.</summary>
    public int MaxEntityTokens { get; init; } = Constants.DefaultMaxEntityTokens;

    /// <summary>Max tokens allocated for relationship context.</summary>
    public int MaxRelationTokens { get; init; } = Constants.DefaultMaxRelationTokens;

    /// <summary>Max total token budget for the entire query context.</summary>
    public int MaxTotalTokens { get; init; } = Constants.DefaultMaxTotalTokens;

    /// <summary>High-level keywords to prioritize in retrieval (overrides LLM extraction when set).</summary>
    public IReadOnlyList<string> HlKeywords { get; init; } = [];

    /// <summary>Low-level keywords to refine retrieval focus (overrides LLM extraction when set).</summary>
    public IReadOnlyList<string> LlKeywords { get; init; } = [];

    /// <summary>Past conversation history, sent to the LLM for context (not used for retrieval).</summary>
    public IReadOnlyList<ChatMessage> ConversationHistory { get; init; } = [];

    /// <summary>Additional user instruction injected into the prompt template.</summary>
    public string? UserPrompt { get; init; }

    /// <summary>Enable reranking for retrieved text chunks (no-op when no reranker is configured).</summary>
    public bool EnableRerank { get; init; } = true;
}
