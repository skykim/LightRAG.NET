namespace LightRAG.Core.Abstractions;

/// <summary>A single entity extracted from text by the LLM. Ported from <c>ExtractedEntity</c> in <c>lightrag/types.py</c>.</summary>
public sealed record ExtractedEntity
{
    public required string EntityName { get; init; }
    public required string EntityType { get; init; }
    public required string EntityDescription { get; init; }
}

/// <summary>A single relationship between two entities. Ported from <c>ExtractedRelationship</c>.</summary>
public sealed record ExtractedRelationship
{
    public required string SourceEntity { get; init; }
    public required string TargetEntity { get; init; }
    public required string RelationshipKeywords { get; init; }
    public required string RelationshipDescription { get; init; }
}

/// <summary>Structured output of entity/relationship extraction. Ported from <c>EntityExtractionResult</c>.</summary>
public sealed record EntityExtractionResult
{
    public List<ExtractedEntity> Entities { get; init; } = [];
    public List<ExtractedRelationship> Relationships { get; init; } = [];
}

/// <summary>A node in a returned knowledge subgraph. Ported from <c>KnowledgeGraphNode</c>.</summary>
public sealed record KnowledgeGraphNode
{
    public required string Id { get; init; }
    public List<string> Labels { get; init; } = [];
    public Dictionary<string, object?> Properties { get; init; } = [];
}

/// <summary>An edge in a returned knowledge subgraph. Ported from <c>KnowledgeGraphEdge</c>.</summary>
public sealed record KnowledgeGraphEdge
{
    public required string Id { get; init; }
    public string? Type { get; init; }
    public required string Source { get; init; }
    public required string Target { get; init; }
    public Dictionary<string, object?> Properties { get; init; } = [];
}

/// <summary>A knowledge subgraph result. Ported from <c>KnowledgeGraph</c> in <c>lightrag/types.py</c>.</summary>
public sealed record KnowledgeGraph
{
    public List<KnowledgeGraphNode> Nodes { get; init; } = [];
    public List<KnowledgeGraphEdge> Edges { get; init; } = [];
    public bool IsTruncated { get; init; }
}
