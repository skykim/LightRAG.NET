using LightRAG.Core.Abstractions;
using LightRAG.Core.Chunking;
using LightRAG.Core.Configuration;
using LightRAG.Core.Extraction;
using LightRAG.Core.Utils;

namespace LightRAG.Core.Pipeline;

/// <summary>A document to ingest.</summary>
/// <param name="Content">Raw document text.</param>
/// <param name="FilePath">Source file path / basename for citations.</param>
public sealed record InputDocument(string Content, string FilePath = "unknown_source");

/// <summary>Options controlling ingestion chunking.</summary>
public sealed record PipelineOptions
{
    public int ChunkTokenSize { get; init; } = Constants.DefaultChunkSize;
    public int ChunkOverlapTokenSize { get; init; } = Constants.DefaultChunkOverlapSize;

    /// <summary>
    /// When true, skip LLM entity/relationship extraction and knowledge-graph construction; only
    /// chunk + embed (vector index). Mirrors Python's <c>PROCESS_OPTION_SKIP_KG</c> ("!"). Enables fast
    /// vector-only ingestion for naive retrieval at scale.
    /// </summary>
    public bool SkipEntityExtraction { get; init; }
}

/// <summary>
/// Document ingestion pipeline, porting the essential loop of <c>lightrag/pipeline.py</c>:
/// enqueue+dedup -&gt; chunk -&gt; store chunks (KV + vector) -&gt; extract -&gt; merge into the graph -&gt;
/// mark processed -&gt; flush. The multi-stage parser/queue machinery is out of scope for this core port.
/// </summary>
public sealed class DocumentPipeline
{
    private readonly ITokenizer _tokenizer;
    private readonly EntityExtractor _extractor;
    private readonly KnowledgeGraphBuilder _builder;
    private readonly IKvStorage _fullDocs;
    private readonly IKvStorage _textChunks;
    private readonly IVectorStorage _chunksVdb;
    private readonly IDocStatusStorage _docStatus;
    private readonly IReadOnlyList<IStorageNameSpace> _allStorages;
    private readonly PipelineOptions _options;

    public DocumentPipeline(
        ITokenizer tokenizer,
        EntityExtractor extractor,
        KnowledgeGraphBuilder builder,
        IKvStorage fullDocs,
        IKvStorage textChunks,
        IVectorStorage chunksVdb,
        IDocStatusStorage docStatus,
        IReadOnlyList<IStorageNameSpace> allStorages,
        PipelineOptions? options = null)
    {
        _tokenizer = tokenizer;
        _extractor = extractor;
        _builder = builder;
        _fullDocs = fullDocs;
        _textChunks = textChunks;
        _chunksVdb = chunksVdb;
        _docStatus = docStatus;
        _allStorages = allStorages;
        _options = options ?? new PipelineOptions();
    }

    /// <summary>Ingest documents: dedup, chunk, extract, merge, persist. Returns the number of docs processed.</summary>
    public async Task<int> InsertAsync(IReadOnlyList<InputDocument> documents, CancellationToken cancellationToken = default)
    {
        var processed = 0;
        foreach (var document in documents)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var content = document.Content.Trim();
            if (content.Length == 0)
            {
                continue;
            }

            var docId = Hashing.ComputeMdHashId(content, "doc-");

            // Dedup: skip documents already fully ingested.
            var missing = await _fullDocs.FilterKeysAsync(new HashSet<string> { docId }, cancellationToken).ConfigureAwait(false);
            if (!missing.Contains(docId))
            {
                continue;
            }

            var nowIso = DateTimeOffset.UtcNow.ToString("o");
            await _docStatus.UpsertAsync(new Dictionary<string, StorageRecord>
            {
                [docId] = NewStatusRecord(content, document.FilePath, DocStatus.Processing, nowIso, nowIso, null),
            }, cancellationToken).ConfigureAwait(false);

            try
            {
                // 1. Chunk.
                var chunks = TokenSizeChunker.Chunk(_tokenizer, content, _options.ChunkTokenSize, _options.ChunkOverlapTokenSize);
                var chunkInputs = new List<ChunkInput>(chunks.Count);
                var chunkKvData = new Dictionary<string, StorageRecord>();
                var chunkVdbData = new Dictionary<string, StorageRecord>();
                var chunkIds = new List<string>();

                foreach (var chunk in chunks)
                {
                    var chunkId = Hashing.ComputeMdHashId(chunk.Content, "chunk-");
                    chunkIds.Add(chunkId);
                    chunkKvData[chunkId] = new StorageRecord
                    {
                        ["tokens"] = chunk.Tokens,
                        ["content"] = chunk.Content,
                        ["full_doc_id"] = docId,
                        ["chunk_order_index"] = chunk.ChunkOrderIndex,
                        ["file_path"] = document.FilePath,
                    };
                    chunkVdbData[chunkId] = new StorageRecord
                    {
                        ["content"] = chunk.Content,
                        ["full_doc_id"] = docId,
                        ["file_path"] = document.FilePath,
                    };
                    chunkInputs.Add(new ChunkInput(chunkId, chunk.Content, document.FilePath));
                }

                // 2. Store chunks (KV + vector).
                await _textChunks.UpsertAsync(chunkKvData, cancellationToken).ConfigureAwait(false);
                await _chunksVdb.UpsertAsync(chunkVdbData, cancellationToken).ConfigureAwait(false);

                if (!_options.SkipEntityExtraction)
                {
                    // 3. Extract entities/relationships per chunk.
                    var extraction = await _extractor.ExtractAsync(chunkInputs, cancellationToken).ConfigureAwait(false);

                    // 4. Merge into the knowledge graph + entity/relationship vectors.
                    await _builder.MergeAsync(extraction, cancellationToken).ConfigureAwait(false);
                }

                // 5. Persist the full document and mark processed.
                await _fullDocs.UpsertAsync(new Dictionary<string, StorageRecord>
                {
                    [docId] = new() { ["content"] = content },
                }, cancellationToken).ConfigureAwait(false);

                await _docStatus.UpsertAsync(new Dictionary<string, StorageRecord>
                {
                    [docId] = NewStatusRecord(content, document.FilePath, DocStatus.Processed, nowIso,
                        DateTimeOffset.UtcNow.ToString("o"), chunkIds),
                }, cancellationToken).ConfigureAwait(false);

                processed++;
            }
            catch (Exception ex)
            {
                await _docStatus.UpsertAsync(new Dictionary<string, StorageRecord>
                {
                    [docId] = NewStatusRecord(content, document.FilePath, DocStatus.Failed, nowIso,
                        DateTimeOffset.UtcNow.ToString("o"), null, ex.Message),
                }, cancellationToken).ConfigureAwait(false);
                await FlushAsync(cancellationToken).ConfigureAwait(false);
                throw;
            }
        }

        // 6. Flush all storages (index_done_callback).
        await FlushAsync(cancellationToken).ConfigureAwait(false);
        return processed;
    }

    private async Task FlushAsync(CancellationToken cancellationToken)
    {
        foreach (var storage in _allStorages)
        {
            await storage.IndexDoneCallbackAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    private static StorageRecord NewStatusRecord(
        string content, string filePath, DocStatus status, string createdAt, string updatedAt,
        IReadOnlyList<string>? chunkIds, string? errorMsg = null)
    {
        var summary = content.Length <= 100 ? content : content[..100];
        var record = new StorageRecord
        {
            ["status"] = status.ToWireValue(),
            ["content_summary"] = summary,
            ["content_length"] = content.Length,
            ["file_path"] = filePath,
            ["created_at"] = createdAt,
            ["updated_at"] = updatedAt,
            ["content_hash"] = Hashing.ComputeArgsHash(content),
        };
        if (chunkIds is not null)
        {
            record["chunks_count"] = chunkIds.Count;
            record["chunks_list"] = chunkIds.Cast<object?>().ToList();
        }
        if (errorMsg is not null)
        {
            record["error_msg"] = errorMsg;
        }
        return record;
    }
}
