namespace LightRAG.Core.Abstractions;

/// <summary>
/// Document processing status, ported from <c>DocStatus</c> in <c>lightrag/base.py</c>.
/// Pipeline order: Pending -&gt; Parsing -&gt; Analyzing -&gt; Processing -&gt; Preprocessed -&gt; Processed | Failed.
/// The core .NET pipeline only emits Pending/Processing/Processed/Failed, but the parser-stage values
/// are carried so on-disk records written by the Python pipeline round-trip without being collapsed.
/// </summary>
public enum DocStatus
{
    Pending,
    Parsing,
    Analyzing,
    Processing,
    Preprocessed,
    Processed,
    Failed,
}

/// <summary>Serialization helpers keeping the on-disk string values identical to Python.</summary>
public static class DocStatusExtensions
{
    /// <summary>All statuses in declaration order (used to zero-seed status counts like Python).</summary>
    public static readonly IReadOnlyList<DocStatus> All =
        (DocStatus[])Enum.GetValues(typeof(DocStatus));

    public static string ToWireValue(this DocStatus status) => status switch
    {
        DocStatus.Pending => "pending",
        DocStatus.Parsing => "parsing",
        DocStatus.Analyzing => "analyzing",
        DocStatus.Processing => "processing",
        DocStatus.Preprocessed => "preprocessed",
        DocStatus.Processed => "processed",
        DocStatus.Failed => "failed",
        _ => throw new ArgumentOutOfRangeException(nameof(status), status, null),
    };

    public static DocStatus FromWireValue(string value) => value switch
    {
        "pending" => DocStatus.Pending,
        "parsing" => DocStatus.Parsing,
        "analyzing" => DocStatus.Analyzing,
        "processing" => DocStatus.Processing,
        "preprocessed" => DocStatus.Preprocessed,
        "processed" => DocStatus.Processed,
        "failed" => DocStatus.Failed,
        _ => throw new ArgumentException($"Unknown doc status: {value}", nameof(value)),
    };
}

/// <summary>
/// Document processing status record, ported from <c>DocProcessingStatus</c> in <c>lightrag/base.py</c>.
/// </summary>
public sealed record DocProcessingStatus
{
    /// <summary>First ~100 chars of document content, used for preview.</summary>
    public required string ContentSummary { get; init; }

    /// <summary>Total length of document content.</summary>
    public required int ContentLength { get; init; }

    /// <summary>Canonical basename of the document (or "unknown_source").</summary>
    public required string FilePath { get; init; }

    /// <summary>Current processing status.</summary>
    public required DocStatus Status { get; init; }

    /// <summary>ISO-8601 timestamp when the document was created.</summary>
    public required string CreatedAt { get; init; }

    /// <summary>ISO-8601 timestamp when the document was last updated.</summary>
    public required string UpdatedAt { get; init; }

    /// <summary>Optional tracking id for monitoring progress.</summary>
    public string? TrackId { get; init; }

    /// <summary>Number of chunks after splitting.</summary>
    public int? ChunksCount { get; init; }

    /// <summary>List of chunk ids associated with this document (used for deletion).</summary>
    public IReadOnlyList<string> ChunksList { get; init; } = [];

    /// <summary>Error message if the document failed.</summary>
    public string? ErrorMsg { get; init; }

    /// <summary>MD5 hash of the underlying content, used for duplicate detection.</summary>
    public string? ContentHash { get; init; }
}
