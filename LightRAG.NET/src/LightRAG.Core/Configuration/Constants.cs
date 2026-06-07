namespace LightRAG.Core.Configuration;

/// <summary>
/// Centralized default configuration constants, ported from <c>lightrag/constants.py</c>.
/// Only the values relevant to the core (ingest + query) path are carried over;
/// server / parser-engine constants are intentionally omitted from this port.
/// </summary>
public static class Constants
{
    // Extraction settings
    public const string DefaultSummaryLanguage = "English";
    public const int DefaultMaxGleaning = 1;
    public const int DefaultEntityNameMaxLength = 256;

    /// <summary>
    /// UTF-8 byte limit for entity identifiers. Some vector backends (e.g. Milvus) validate
    /// VARCHAR max_length in BYTES, so a CJK name within the character limit can still overflow
    /// (256 Chinese chars ~= 694 bytes &gt; 512). Ported from <c>DEFAULT_ENTITY_NAME_MAX_BYTES</c>.
    /// </summary>
    public const int DefaultEntityNameMaxBytes = 512;
    public const int DefaultMaxExtractionRecords = 100;
    public const int DefaultMaxExtractionEntities = 40;

    // Description merge / summary thresholds
    public const int DefaultForceLlmSummaryOnMerge = 8;
    public const int DefaultSummaryMaxTokens = 1200;
    public const int DefaultSummaryLengthRecommended = 600;
    public const int DefaultSummaryContextSize = 12000;
    public const int DefaultMaxExtractInputTokens = 20480;

    /// <summary>Separator for description, source_id and relation-key fields. Must not change after data is inserted.</summary>
    public const string GraphFieldSep = "<SEP>";

    // Query and retrieval defaults
    public const int DefaultTopK = 40;
    public const int DefaultChunkTopK = 20;
    public const int DefaultMaxEntityTokens = 6000;
    public const int DefaultMaxRelationTokens = 8000;
    public const int DefaultMaxTotalTokens = 30000;
    public const float DefaultCosineThreshold = 0.2f;
    public const int DefaultRelatedChunkNumber = 5;

    /// <summary>KG chunk-selection strategy: "VECTOR" (cosine similarity) or "WEIGHT" (weighted polling).</summary>
    public const string DefaultKgChunkPickMethod = "VECTOR";

    /// <summary>Minimum rerank score to retain a chunk when reranking is enabled.</summary>
    public const double DefaultMinRerankScore = 0.5;

    // Source-id limits
    public const int DefaultMaxSourceIdsPerEntity = 200;
    public const int DefaultMaxSourceIdsPerRelation = 200;
    public const int DefaultMaxFilePaths = 75;

    // Chunking defaults
    public const int DefaultChunkSize = 1200;
    public const int DefaultChunkOverlapSize = 100;

    // LLM defaults
    public const double DefaultTemperature = 1.0;

    // Async / concurrency defaults
    public const int DefaultMaxAsync = 4;
    public const int DefaultMaxParallelInsert = 3;
    public const int DefaultEmbeddingFuncMaxAsync = 8;
    public const int DefaultEmbeddingBatchNum = 10;

    // Priority levels (lower runs first within a role queue)
    public const int DefaultQueryPriority = 5;
    public const int DefaultSummaryPriority = 8;
    public const int DefaultProcessingPriority = 10;

    // Timeouts (seconds)
    public const int DefaultLlmTimeout = 180;
    public const int DefaultEmbeddingTimeout = 30;

    // Ollama emulation defaults
    public const string DefaultOllamaModelName = "lightrag";
    public const string DefaultOllamaModelTag = "latest";
}
