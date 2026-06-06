using LightRAG.Core.Abstractions;
using LightRAG.Core.Configuration;
using LightRAG.Core.Llm;
using LightRAG.Core.Prompts;
using LightRAG.Core.Utils;
using Newtonsoft.Json.Linq;

namespace LightRAG.Core.Query;

/// <summary>Static configuration for the query engine.</summary>
public sealed record QueryOptions
{
    public string Language { get; init; } = Constants.DefaultSummaryLanguage;
}

/// <summary>
/// Knowledge-graph + vector retrieval and answer generation. Ports the query path of
/// <c>lightrag/operate.py</c> (<c>kg_query</c>, <c>extract_keywords_only</c>, <c>_build_query_context</c>,
/// <c>_get_node_data</c>/<c>_get_edge_data</c>/<c>_get_vector_context</c>, <c>naive_query</c>) into a
/// pragmatic single-class pipeline: keywords -&gt; retrieve -&gt; truncate -&gt; build context -&gt; answer.
/// </summary>
public sealed class QueryEngine
{
    private readonly IGraphStorage _graph;
    private readonly IVectorStorage _entitiesVdb;
    private readonly IVectorStorage _relationshipsVdb;
    private readonly IVectorStorage _chunksVdb;
    private readonly IKvStorage _textChunks;
    private readonly ITokenizer _tokenizer;
    private readonly ILlmCaller _queryLlm;
    private readonly ILlmCaller _keywordLlm;
    private readonly QueryOptions _options;

    public QueryEngine(
        IGraphStorage graph,
        IVectorStorage entitiesVdb,
        IVectorStorage relationshipsVdb,
        IVectorStorage chunksVdb,
        IKvStorage textChunks,
        ITokenizer tokenizer,
        ILlmCaller queryLlm,
        ILlmCaller keywordLlm,
        QueryOptions? options = null)
    {
        _graph = graph;
        _entitiesVdb = entitiesVdb;
        _relationshipsVdb = relationshipsVdb;
        _chunksVdb = chunksVdb;
        _textChunks = textChunks;
        _tokenizer = tokenizer;
        _queryLlm = queryLlm;
        _keywordLlm = keywordLlm;
        _options = options ?? new QueryOptions();
    }

    public async Task<QueryResult> QueryAsync(
        string query,
        QueryParam? param = null,
        string? systemPrompt = null,
        CancellationToken cancellationToken = default)
    {
        param ??= new QueryParam();
        if (string.IsNullOrWhiteSpace(query))
        {
            return new QueryResult { Content = PromptTemplates.FailResponse };
        }

        if (param.Mode == QueryMode.Bypass)
        {
            var bypass = await _queryLlm.CompleteAsync(query, systemPrompt, param.ConversationHistory,
                cacheType: "query", cancellationToken: cancellationToken).ConfigureAwait(false);
            return new QueryResult { Content = bypass };
        }

        var context = param.Mode == QueryMode.Naive
            ? await BuildNaiveContextAsync(query, param, cancellationToken).ConfigureAwait(false)
            : await BuildKgContextAsync(query, param, cancellationToken).ConfigureAwait(false);

        if (context is null)
        {
            return new QueryResult { Content = PromptTemplates.FailResponse };
        }

        if (param.OnlyNeedContext)
        {
            return new QueryResult { Content = context.Context, RawData = context.RawData };
        }

        var template = param.Mode == QueryMode.Naive ? PromptTemplates.NaiveRagResponse : PromptTemplates.RagResponse;
        template = systemPrompt ?? template;
        var sysPrompt = PromptRenderer.Render(template,
            ("response_type", string.IsNullOrEmpty(param.ResponseType) ? "Multiple Paragraphs" : param.ResponseType),
            ("user_prompt", string.IsNullOrEmpty(param.UserPrompt) ? "n/a" : param.UserPrompt!),
            ("context_data", context.Context),
            ("content_data", context.Context));

        if (param.OnlyNeedPrompt)
        {
            return new QueryResult { Content = $"{sysPrompt}\n\n---User Query---\n\n{query}", RawData = context.RawData };
        }

        var answer = await _queryLlm.CompleteAsync(query, sysPrompt, param.ConversationHistory,
            cacheType: "query", cancellationToken: cancellationToken).ConfigureAwait(false);
        return new QueryResult { Content = answer, RawData = context.RawData };
    }

    // ---- keyword extraction ----

    public async Task<(IReadOnlyList<string> High, IReadOnlyList<string> Low)> ExtractKeywordsAsync(
        string query, CancellationToken cancellationToken = default)
    {
        var examples = string.Join("\n", PromptTemplates.KeywordsExtractionExamples);
        var prompt = PromptRenderer.Render(PromptTemplates.KeywordsExtraction,
            ("query", query),
            ("examples", examples),
            ("language", _options.Language));

        var response = await _keywordLlm.CompleteAsync(prompt, jsonMode: true,
            cacheType: "keywords", cancellationToken: cancellationToken).ConfigureAwait(false);

        var token = JsonRepair.TryParse(TextUtils.RemoveThinkTags(response));
        if (token is not JObject root)
        {
            return ([], []);
        }
        return (ReadStringArray(root, "high_level_keywords"),
                ReadStringArray(root, "low_level_keywords"));
    }

    // ---- naive (chunk-only) retrieval ----

    private async Task<QueryContextResult?> BuildNaiveContextAsync(string query, QueryParam param, CancellationToken cancellationToken)
    {
        var topK = param.ChunkTopK > 0 ? param.ChunkTopK : param.TopK;
        var results = await _chunksVdb.QueryAsync(query, topK, cancellationToken: cancellationToken).ConfigureAwait(false);
        var chunks = results
            .Where(r => r.ContainsKey("content"))
            .Select(r => new ChunkEntry(r.GetString("content")!, r.GetString("file_path") ?? "unknown_source"))
            .ToList();

        if (chunks.Count == 0)
        {
            return null;
        }

        chunks = TruncateChunks(chunks, param.MaxTotalTokens);
        var (chunksStr, referenceStr) = SerializeChunks(chunks);
        var context = PromptRenderer.Render(PromptTemplates.NaiveQueryContext,
            ("text_chunks_str", chunksStr),
            ("reference_list_str", referenceStr));

        return new QueryContextResult(context, BuildRawData([], [], chunks));
    }

    // ---- knowledge-graph retrieval (local / global / hybrid / mix) ----

    private async Task<QueryContextResult?> BuildKgContextAsync(string query, QueryParam param, CancellationToken cancellationToken)
    {
        IReadOnlyList<string> hl = param.HlKeywords;
        IReadOnlyList<string> ll = param.LlKeywords;
        if (hl.Count == 0 && ll.Count == 0)
        {
            (hl, ll) = await ExtractKeywordsAsync(query, cancellationToken).ConfigureAwait(false);
        }
        if (hl.Count == 0 && ll.Count == 0 && query.Length < 50)
        {
            ll = [query]; // fallback: use the query itself as a low-level keyword
        }

        var useLocal = param.Mode is QueryMode.Local or QueryMode.Hybrid or QueryMode.Mix;
        var useGlobal = param.Mode is QueryMode.Global or QueryMode.Hybrid or QueryMode.Mix;

        var entities = new List<EntityEntry>();
        var relations = new List<RelationEntry>();

        if (useLocal && ll.Count > 0)
        {
            var hits = await _entitiesVdb.QueryAsync(string.Join(", ", ll), param.TopK, cancellationToken: cancellationToken).ConfigureAwait(false);
            foreach (var hit in hits)
            {
                var name = hit.GetString("entity_name");
                if (name is null)
                {
                    continue;
                }
                var node = await _graph.GetNodeAsync(name, cancellationToken).ConfigureAwait(false);
                entities.Add(new EntityEntry(
                    name,
                    node?.GetString("entity_type") ?? hit.GetString("entity_type") ?? "UNKNOWN",
                    node?.GetString("description") ?? string.Empty,
                    node?.GetString("file_path") ?? hit.GetString("file_path") ?? "unknown_source",
                    node?.GetString("source_id") ?? hit.GetString("source_id")));
            }
        }

        if (useGlobal && hl.Count > 0)
        {
            var hits = await _relationshipsVdb.QueryAsync(string.Join(", ", hl), param.TopK, cancellationToken: cancellationToken).ConfigureAwait(false);
            foreach (var hit in hits)
            {
                relations.Add(new RelationEntry(
                    hit.GetString("src_id") ?? string.Empty,
                    hit.GetString("tgt_id") ?? string.Empty,
                    hit.GetString("keywords") ?? string.Empty,
                    hit.GetString("description") ?? string.Empty,
                    hit.GetString("file_path") ?? "unknown_source",
                    hit.GetString("source_id")));
            }
        }

        // Truncate entities / relations by their token budgets.
        entities = TextUtils.TruncateListByTokenSize(entities, e => e.Description, param.MaxEntityTokens, _tokenizer).ToList();
        relations = TextUtils.TruncateListByTokenSize(relations, r => r.Description, param.MaxRelationTokens, _tokenizer).ToList();

        // Gather source chunks referenced by surviving entities/relations.
        var chunkIds = new List<string>();
        var seen = new HashSet<string>();
        foreach (var sid in entities.SelectMany(e => SplitSep(e.SourceId)).Concat(relations.SelectMany(r => SplitSep(r.SourceId))))
        {
            if (seen.Add(sid))
            {
                chunkIds.Add(sid);
            }
        }

        var chunks = new List<ChunkEntry>();
        if (chunkIds.Count > 0)
        {
            var records = await _textChunks.GetByIdsAsync(chunkIds, cancellationToken).ConfigureAwait(false);
            foreach (var record in records)
            {
                var content = record?.GetString("content");
                if (content is not null)
                {
                    chunks.Add(new ChunkEntry(content, record!.GetString("file_path") ?? "unknown_source"));
                }
            }
        }

        // Mix mode: add direct vector chunks.
        if (param.Mode == QueryMode.Mix)
        {
            var topK = param.ChunkTopK > 0 ? param.ChunkTopK : param.TopK;
            var vectorHits = await _chunksVdb.QueryAsync(query, topK, cancellationToken: cancellationToken).ConfigureAwait(false);
            foreach (var hit in vectorHits)
            {
                var content = hit.GetString("content");
                if (content is not null && !chunks.Any(c => c.Content == content))
                {
                    chunks.Add(new ChunkEntry(content, hit.GetString("file_path") ?? "unknown_source"));
                }
            }
        }

        if (entities.Count == 0 && relations.Count == 0 && chunks.Count == 0)
        {
            return null;
        }

        chunks = TruncateChunks(chunks, param.MaxTotalTokens);

        var entitiesStr = SerializeEntities(entities);
        var relationsStr = SerializeRelations(relations);
        var (chunksStr, referenceStr) = SerializeChunks(chunks);

        var context = PromptRenderer.Render(PromptTemplates.KgQueryContext,
            ("entities_str", entitiesStr),
            ("relations_str", relationsStr),
            ("text_chunks_str", chunksStr),
            ("reference_list_str", referenceStr));

        return new QueryContextResult(context, BuildRawData(entities, relations, chunks));
    }

    // ---- serialization helpers ----

    private string SerializeEntities(List<EntityEntry> entities)
    {
        var list = entities.Select((e, i) => new Dictionary<string, object?>
        {
            ["id"] = i + 1,
            ["entity"] = e.Name,
            ["type"] = e.Type,
            ["description"] = e.Description,
            ["file_path"] = e.FilePath,
        });
        return JsonHelper.Serialize(list);
    }

    private string SerializeRelations(List<RelationEntry> relations)
    {
        var list = relations.Select((r, i) => new Dictionary<string, object?>
        {
            ["id"] = i + 1,
            ["entity1"] = r.Src,
            ["entity2"] = r.Tgt,
            ["keywords"] = r.Keywords,
            ["description"] = r.Description,
            ["file_path"] = r.FilePath,
        });
        return JsonHelper.Serialize(list);
    }

    private (string ChunksStr, string ReferenceStr) SerializeChunks(List<ChunkEntry> chunks)
    {
        var fileToRef = new Dictionary<string, int>();
        var references = new List<string>();
        var chunkList = new List<Dictionary<string, object?>>();

        foreach (var chunk in chunks)
        {
            if (!fileToRef.TryGetValue(chunk.FilePath, out var refId))
            {
                refId = references.Count + 1;
                fileToRef[chunk.FilePath] = refId;
                references.Add($"[{refId}] {chunk.FilePath}");
            }
            chunkList.Add(new Dictionary<string, object?>
            {
                ["reference_id"] = refId,
                ["content"] = chunk.Content,
                ["file_path"] = chunk.FilePath,
            });
        }

        return (JsonHelper.Serialize(chunkList), string.Join("\n", references));
    }

    private List<ChunkEntry> TruncateChunks(List<ChunkEntry> chunks, int maxTokens)
        => TextUtils.TruncateListByTokenSize(chunks, c => c.Content, maxTokens, _tokenizer).ToList();

    private static IReadOnlyDictionary<string, object?> BuildRawData(
        List<EntityEntry> entities, List<RelationEntry> relations, List<ChunkEntry> chunks)
    {
        var references = chunks.Select(c => c.FilePath).Distinct()
            .Select((fp, i) => (object?)new Dictionary<string, object?> { ["reference_id"] = (i + 1).ToString(), ["file_path"] = fp })
            .ToList();
        return new Dictionary<string, object?>
        {
            ["data"] = new Dictionary<string, object?>
            {
                ["entities"] = entities.Select(e => (object?)new Dictionary<string, object?> { ["entity"] = e.Name, ["type"] = e.Type, ["description"] = e.Description }).ToList(),
                ["relationships"] = relations.Select(r => (object?)new Dictionary<string, object?> { ["src"] = r.Src, ["tgt"] = r.Tgt, ["description"] = r.Description }).ToList(),
                ["chunks"] = chunks.Select(c => (object?)new Dictionary<string, object?> { ["content"] = c.Content, ["file_path"] = c.FilePath }).ToList(),
                ["references"] = references,
            },
        };
    }

    private static IReadOnlyList<string> ReadStringArray(JObject obj, string property)
    {
        if (obj[property] is not JArray arr)
        {
            return [];
        }
        return arr
            .Where(e => e.Type == JTokenType.String)
            .Select(e => e.Value<string>()!)
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .ToList();
    }

    private static IEnumerable<string> SplitSep(string? value)
        => string.IsNullOrEmpty(value) ? [] : value!.Split(new[] { Constants.GraphFieldSep }, StringSplitOptions.RemoveEmptyEntries);

    private sealed record EntityEntry(string Name, string Type, string Description, string FilePath, string? SourceId);
    private sealed record RelationEntry(string Src, string Tgt, string Keywords, string Description, string FilePath, string? SourceId);
    private sealed record ChunkEntry(string Content, string FilePath);
}
