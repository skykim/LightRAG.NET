namespace LightRAG.Core.Extraction;

/// <summary>An entity extracted from a single chunk. Mirrors the dict built by <c>_handle_single_entity_extraction</c>.</summary>
public sealed record ExtractedNode
{
    public required string EntityName { get; init; }
    public required string EntityType { get; init; }
    public required string Description { get; init; }
    public required string SourceId { get; init; }
    public required string FilePath { get; init; }
    public long Timestamp { get; init; }
}

/// <summary>A relationship extracted from a single chunk. Mirrors the dict built by <c>_handle_single_relationship_extraction</c>.</summary>
public sealed record ExtractedEdge
{
    public required string SrcId { get; init; }
    public required string TgtId { get; init; }
    public double Weight { get; init; } = 1.0;
    public required string Description { get; init; }
    public required string Keywords { get; init; }
    public required string SourceId { get; init; }
    public required string FilePath { get; init; }
    public long Timestamp { get; init; }
}

/// <summary>
/// The per-chunk extraction output: entity name -&gt; candidate nodes, and (src,tgt) -&gt; candidate edges.
/// Multiple candidates per key accumulate before the knowledge-graph merge step.
/// </summary>
public sealed class ChunkExtractionResult
{
    public Dictionary<string, List<ExtractedNode>> Nodes { get; } = new();
    public Dictionary<(string Src, string Tgt), List<ExtractedEdge>> Edges { get; } = new();
}
