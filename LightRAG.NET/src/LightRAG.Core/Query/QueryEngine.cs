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

    /// <summary>KG chunk-selection strategy: "VECTOR" (cosine similarity) or "WEIGHT" (weighted polling).</summary>
    public string KgChunkPickMethod { get; init; } = Constants.DefaultKgChunkPickMethod;

    /// <summary>Per-entity/relation cap on related chunks (the weighted-polling top quota).</summary>
    public int RelatedChunkNumber { get; init; } = Constants.DefaultRelatedChunkNumber;

    /// <summary>Minimum rerank score to keep a chunk (only applied when a reranker is configured).</summary>
    public double MinRerankScore { get; init; } = Constants.DefaultMinRerankScore;

    /// <summary>
    /// Optional reranker: given the query and chunk contents, returns one score per chunk.
    /// When null (the default), reranking is a no-op — matching Python with no RERANK binding.
    /// </summary>
    public Func<string, IReadOnlyList<string>, CancellationToken, Task<IReadOnlyList<double>>>? Reranker { get; init; }
}

/// <summary>
/// Knowledge-graph + vector retrieval and answer generation. Ports the query path of
/// <c>lightrag/operate.py</c> (<c>kg_query</c>, <c>extract_keywords_only</c>, <c>_perform_kg_search</c>,
/// <c>_get_node_data</c>/<c>_get_edge_data</c> with their 1-hop graph expansion, the weighted-polling /
/// vector-similarity chunk selection, round-robin merges, <c>process_chunks_unified</c>, and
/// <c>_build_context_str</c> with its dynamic token budget).
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

    private const int BufferTokens = 200; // reserved for reference list + safety, matches Python

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
            ? await BuildNaiveContextAsync(query, param, systemPrompt, cancellationToken).ConfigureAwait(false)
            : await BuildKgContextAsync(query, param, systemPrompt, cancellationToken).ConfigureAwait(false);

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
        // Python prefixes a non-empty user_prompt with two newlines; otherwise "n/a".
        var userPrompt = string.IsNullOrEmpty(param.UserPrompt) ? "n/a" : $"\n\n{param.UserPrompt}";
        var sysPrompt = PromptRenderer.Render(template,
            ("response_type", string.IsNullOrEmpty(param.ResponseType) ? "Multiple Paragraphs" : param.ResponseType),
            ("user_prompt", userPrompt),
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

    private async Task<QueryContextResult?> BuildNaiveContextAsync(string query, QueryParam param, string? systemPrompt, CancellationToken cancellationToken)
    {
        var vectorChunks = await GetVectorContextAsync(query, param, cancellationToken).ConfigureAwait(false);
        if (vectorChunks.Count == 0)
        {
            return null;
        }

        // Dynamic chunk budget: total - (system-prompt overhead + query + buffer).
        var template = systemPrompt ?? PromptTemplates.NaiveRagResponse;
        var sysOverhead = _tokenizer.CountTokens(PromptRenderer.Render(template,
            ("response_type", string.IsNullOrEmpty(param.ResponseType) ? "Multiple Paragraphs" : param.ResponseType),
            ("user_prompt", string.IsNullOrEmpty(param.UserPrompt) ? "n/a" : $"\n\n{param.UserPrompt}"),
            ("content_data", ""),
            ("context_data", "")));
        var available = param.MaxTotalTokens - (sysOverhead + _tokenizer.CountTokens(query) + BufferTokens);

        var processed = await ProcessChunksUnifiedAsync(query, vectorChunks, param, available, cancellationToken).ConfigureAwait(false);
        var (refList, refIds) = GenerateReferenceList(processed);

        var (chunksStr, referenceStr) = SerializeChunks(processed, refIds, refList);
        var context = PromptRenderer.Render(PromptTemplates.NaiveQueryContext,
            ("text_chunks_str", chunksStr),
            ("reference_list_str", referenceStr));

        return new QueryContextResult(context, BuildRawData([], [], processed, refList));
    }

    // ---- knowledge-graph retrieval (local / global / hybrid / mix) ----

    private async Task<QueryContextResult?> BuildKgContextAsync(string query, QueryParam param, string? systemPrompt, CancellationToken cancellationToken)
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

        // Stage 1: pure search (entities, relations, vector chunks) with 1-hop graph expansion.
        var search = await PerformKgSearchAsync(query, hl, ll, param, cancellationToken).ConfigureAwait(false);

        if (search.Entities.Count == 0 && search.Relations.Count == 0)
        {
            if (param.Mode != QueryMode.Mix || search.VectorChunks.Count == 0)
            {
                return null;
            }
        }

        // Stage 2: token truncation of entities/relations -> context dicts + filtered survivors.
        var trunc = ApplyTokenTruncation(search.Entities, search.Relations, param);

        // Stage 3: merge chunks from vector + entity + relation sources (round-robin, deduped).
        var merged = await MergeAllChunksAsync(query, trunc.FilteredEntities, trunc.FilteredRelations, search.VectorChunks, param, cancellationToken).ConfigureAwait(false);

        if (merged.Count == 0 && trunc.EntitiesContext.Count == 0 && trunc.RelationsContext.Count == 0)
        {
            return null;
        }

        // Stage 4: build the final context string with a dynamic chunk token budget.
        var entitiesStr = string.Join("\n", trunc.EntitiesContext.Select(e => JsonHelper.SerializeLine(
            ("entity", (object?)e.Name), ("type", e.Type), ("description", e.Description), ("created_at", e.CreatedAt), ("file_path", e.FilePath))));
        var relationsStr = string.Join("\n", trunc.RelationsContext.Select(r => JsonHelper.SerializeLine(
            ("entity1", (object?)r.Entity1), ("entity2", r.Entity2), ("description", r.Description), ("created_at", r.CreatedAt), ("file_path", r.FilePath))));

        // Preliminary KG context tokens (chunks/reference empty), and system-prompt overhead.
        var preKg = PromptRenderer.Render(PromptTemplates.KgQueryContext,
            ("entities_str", entitiesStr), ("relations_str", relationsStr), ("text_chunks_str", ""), ("reference_list_str", ""));
        var kgContextTokens = _tokenizer.CountTokens(preKg);

        var sysTemplate = systemPrompt ?? PromptTemplates.RagResponse;
        var sysOverhead = _tokenizer.CountTokens(PromptRenderer.Render(sysTemplate,
            ("response_type", string.IsNullOrEmpty(param.ResponseType) ? "Multiple Paragraphs" : param.ResponseType),
            ("user_prompt", param.UserPrompt ?? ""),
            ("context_data", ""), ("content_data", "")));
        var available = param.MaxTotalTokens - (sysOverhead + kgContextTokens + _tokenizer.CountTokens(query) + BufferTokens);

        var processed = await ProcessChunksUnifiedAsync(query, merged, param, available, cancellationToken).ConfigureAwait(false);
        var (refList, refIds) = GenerateReferenceList(processed);
        var (chunksStr, referenceStr) = SerializeChunks(processed, refIds, refList);

        if (trunc.EntitiesContext.Count == 0 && trunc.RelationsContext.Count == 0 && processed.Count == 0)
        {
            return null;
        }

        var context = PromptRenderer.Render(PromptTemplates.KgQueryContext,
            ("entities_str", entitiesStr),
            ("relations_str", relationsStr),
            ("text_chunks_str", chunksStr),
            ("reference_list_str", referenceStr));

        return new QueryContextResult(context, BuildRawData(trunc.EntitiesContext, trunc.RelationsContext, processed, refList));
    }

    // ---- Stage 1: pure KG search ----

    private sealed record KgSearchResult(List<EntityData> Entities, List<RelationData> Relations, List<ChunkData> VectorChunks);

    private async Task<KgSearchResult> PerformKgSearchAsync(string query, IReadOnlyList<string> hl, IReadOnlyList<string> ll, QueryParam param, CancellationToken ct)
    {
        var useLocal = param.Mode is QueryMode.Local or QueryMode.Hybrid or QueryMode.Mix;
        var useGlobal = param.Mode is QueryMode.Global or QueryMode.Hybrid or QueryMode.Mix;

        var localEntities = new List<EntityData>();
        var localRelations = new List<RelationData>();
        var globalEntities = new List<EntityData>();
        var globalRelations = new List<RelationData>();
        var vectorChunks = new List<ChunkData>();

        if (useLocal && ll.Count > 0)
        {
            (localEntities, localRelations) = await GetNodeDataAsync(string.Join(", ", ll), param, ct).ConfigureAwait(false);
        }
        if (useGlobal && hl.Count > 0)
        {
            (globalRelations, globalEntities) = await GetEdgeDataAsync(string.Join(", ", hl), param, ct).ConfigureAwait(false);
        }
        if (param.Mode == QueryMode.Mix)
        {
            vectorChunks = await GetVectorContextAsync(query, param, ct).ConfigureAwait(false);
        }

        // Round-robin merge entities (local then global), dedup by name.
        var entities = new List<EntityData>();
        var seenEntities = new HashSet<string>();
        var maxLen = Math.Max(localEntities.Count, globalEntities.Count);
        for (var i = 0; i < maxLen; i++)
        {
            if (i < localEntities.Count && seenEntities.Add(localEntities[i].Name))
            {
                entities.Add(localEntities[i]);
            }
            if (i < globalEntities.Count && seenEntities.Add(globalEntities[i].Name))
            {
                entities.Add(globalEntities[i]);
            }
        }

        // Round-robin merge relations (local then global), dedup by sorted (src, tgt).
        var relations = new List<RelationData>();
        var seenRelations = new HashSet<(string, string)>();
        maxLen = Math.Max(localRelations.Count, globalRelations.Count);
        for (var i = 0; i < maxLen; i++)
        {
            if (i < localRelations.Count && seenRelations.Add(SortedPair(localRelations[i].Src, localRelations[i].Tgt)))
            {
                relations.Add(localRelations[i]);
            }
            if (i < globalRelations.Count && seenRelations.Add(SortedPair(globalRelations[i].Src, globalRelations[i].Tgt)))
            {
                relations.Add(globalRelations[i]);
            }
        }

        return new KgSearchResult(entities, relations, vectorChunks);
    }

    // local mode: entities VDB -> nodes; then their incident edges (1-hop expansion).
    private async Task<(List<EntityData>, List<RelationData>)> GetNodeDataAsync(string keywords, QueryParam param, CancellationToken ct)
    {
        var hits = await _entitiesVdb.QueryAsync(keywords, param.TopK, cancellationToken: ct).ConfigureAwait(false);
        if (hits.Count == 0)
        {
            return ([], []);
        }

        var names = hits.Select(h => h.GetString("entity_name")).Where(n => n is not null).Cast<string>().ToList();
        var nodes = await _graph.GetNodesBatchAsync(names, ct).ConfigureAwait(false);
        var degrees = await _graph.NodeDegreesBatchAsync(names, ct).ConfigureAwait(false);

        var entities = new List<EntityData>();
        foreach (var name in names)
        {
            if (!nodes.TryGetValue(name, out var node))
            {
                continue;
            }
            entities.Add(new EntityData(
                name,
                node.GetString("entity_type") ?? "UNKNOWN",
                node.GetString("description") ?? "UNKNOWN",
                node.GetString("file_path") ?? "unknown_source",
                node.GetString("source_id"),
                GetCreatedAt(node),
                degrees.TryGetValue(name, out var d) ? d : 0));
        }

        var relations = await FindMostRelatedEdgesFromEntitiesAsync(entities, ct).ConfigureAwait(false);
        return (entities, relations);
    }

    // 1-hop: incident edges of the entity nodes, sorted by (rank, weight) desc.
    private async Task<List<RelationData>> FindMostRelatedEdgesFromEntitiesAsync(List<EntityData> entities, CancellationToken ct)
    {
        var names = entities.Select(e => e.Name).ToList();
        var edgesByNode = await _graph.GetNodesEdgesBatchAsync(names, ct).ConfigureAwait(false);

        var seen = new HashSet<(string, string)>();
        var orderedPairs = new List<(string Src, string Tgt)>();
        foreach (var name in names)
        {
            if (!edgesByNode.TryGetValue(name, out var edges))
            {
                continue;
            }
            foreach (var e in edges)
            {
                var key = SortedPair(e.Source, e.Target);
                if (seen.Add(key))
                {
                    orderedPairs.Add(key);
                }
            }
        }

        var relations = new List<RelationData>();
        foreach (var pair in orderedPairs)
        {
            var edge = await _graph.GetEdgeAsync(pair.Src, pair.Tgt, ct).ConfigureAwait(false);
            if (edge is null)
            {
                continue;
            }
            var rank = await _graph.EdgeDegreeAsync(pair.Src, pair.Tgt, ct).ConfigureAwait(false);
            relations.Add(new RelationData(
                pair.Src, pair.Tgt,
                edge.GetString("keywords") ?? string.Empty,
                edge.GetString("description") ?? string.Empty,
                edge.GetString("file_path") ?? "unknown_source",
                edge.GetString("source_id"),
                GetCreatedAt(edge),
                GetWeight(edge),
                rank));
        }

        return relations
            .OrderByDescending(r => r.Rank)
            .ThenByDescending(r => r.Weight)
            .ToList();
    }

    // global mode: relationships VDB -> edges; then their endpoint entities (1-hop expansion).
    private async Task<(List<RelationData>, List<EntityData>)> GetEdgeDataAsync(string keywords, QueryParam param, CancellationToken ct)
    {
        var hits = await _relationshipsVdb.QueryAsync(keywords, param.TopK, cancellationToken: ct).ConfigureAwait(false);
        if (hits.Count == 0)
        {
            return ([], []);
        }

        var relations = new List<RelationData>();
        foreach (var hit in hits)
        {
            var src = hit.GetString("src_id");
            var tgt = hit.GetString("tgt_id");
            if (src is null || tgt is null)
            {
                continue;
            }
            var edge = await _graph.GetEdgeAsync(src, tgt, ct).ConfigureAwait(false);
            if (edge is null)
            {
                continue;
            }
            relations.Add(new RelationData(
                src, tgt,
                edge.GetString("keywords") ?? string.Empty,
                edge.GetString("description") ?? string.Empty,
                edge.GetString("file_path") ?? "unknown_source",
                edge.GetString("source_id"),
                GetCreatedAt(edge),
                GetWeight(edge),
                0));
        }

        var entities = await FindMostRelatedEntitiesFromRelationshipsAsync(relations, ct).ConfigureAwait(false);
        return (relations, entities);
    }

    // 1-hop: endpoint entities of the relations, in first-seen order (src then tgt).
    private async Task<List<EntityData>> FindMostRelatedEntitiesFromRelationshipsAsync(List<RelationData> relations, CancellationToken ct)
    {
        var names = new List<string>();
        var seen = new HashSet<string>();
        foreach (var r in relations)
        {
            if (seen.Add(r.Src)) { names.Add(r.Src); }
            if (seen.Add(r.Tgt)) { names.Add(r.Tgt); }
        }

        var nodes = await _graph.GetNodesBatchAsync(names, ct).ConfigureAwait(false);
        var entities = new List<EntityData>();
        foreach (var name in names)
        {
            if (!nodes.TryGetValue(name, out var node))
            {
                continue;
            }
            entities.Add(new EntityData(
                name,
                node.GetString("entity_type") ?? "UNKNOWN",
                node.GetString("description") ?? "UNKNOWN",
                node.GetString("file_path") ?? "unknown_source",
                node.GetString("source_id"),
                GetCreatedAt(node),
                0));
        }
        return entities;
    }

    private async Task<List<ChunkData>> GetVectorContextAsync(string query, QueryParam param, CancellationToken ct)
    {
        var topK = param.ChunkTopK > 0 ? param.ChunkTopK : param.TopK;
        var results = await _chunksVdb.QueryAsync(query, topK, cancellationToken: ct).ConfigureAwait(false);
        var chunks = new List<ChunkData>();
        foreach (var r in results)
        {
            var content = r.GetString("content");
            if (content is not null)
            {
                chunks.Add(new ChunkData(content, r.GetString("file_path") ?? "unknown_source", r.GetString("id") ?? string.Empty));
            }
        }
        return chunks;
    }

    // ---- Stage 2: token truncation of entities / relations ----

    private sealed record EntityContext(string Name, string Type, string Description, string CreatedAt, string FilePath);
    private sealed record RelationContext(string Entity1, string Entity2, string Description, string CreatedAt, string FilePath);
    private sealed record TruncationResult(
        List<EntityContext> EntitiesContext, List<RelationContext> RelationsContext,
        List<EntityData> FilteredEntities, List<RelationData> FilteredRelations);

    private TruncationResult ApplyTokenTruncation(List<EntityData> entities, List<RelationData> relations, QueryParam param)
    {
        var entitiesContext = entities.Select(e => new EntityContext(
            e.Name, e.Type, e.Description, FormatCreatedAt(e.CreatedAt), e.FilePath)).ToList();
        var relationsContext = relations.Select(r => new RelationContext(
            r.Src, r.Tgt, r.Description, FormatCreatedAt(r.CreatedAt), r.FilePath)).ToList();

        // Token budget counts only the fields the LLM scores on (entity/type/description), not file_path/created_at.
        entitiesContext = TextUtils.TruncateListByTokenSize(entitiesContext,
            e => JsonHelper.SerializeLine(("entity", e.Name), ("type", e.Type), ("description", e.Description)),
            param.MaxEntityTokens, _tokenizer).ToList();
        relationsContext = TextUtils.TruncateListByTokenSize(relationsContext,
            r => JsonHelper.SerializeLine(("entity1", r.Entity1), ("entity2", r.Entity2), ("description", r.Description)),
            param.MaxRelationTokens, _tokenizer).ToList();

        var keptEntityNames = entitiesContext.Select(e => e.Name).ToHashSet();
        var filteredEntities = new List<EntityData>();
        var seenNodes = new HashSet<string>();
        foreach (var e in entities)
        {
            if (keptEntityNames.Contains(e.Name) && seenNodes.Add(e.Name))
            {
                filteredEntities.Add(e);
            }
        }

        var keptPairs = relationsContext.Select(r => (r.Entity1, r.Entity2)).ToHashSet();
        var filteredRelations = new List<RelationData>();
        var seenEdges = new HashSet<(string, string)>();
        foreach (var r in relations)
        {
            var pair = (r.Src, r.Tgt);
            if (keptPairs.Contains(pair) && seenEdges.Add(pair))
            {
                filteredRelations.Add(r);
            }
        }

        return new TruncationResult(entitiesContext, relationsContext, filteredEntities, filteredRelations);
    }

    // ---- Stage 3: merge chunks ----

    private async Task<List<ChunkData>> MergeAllChunksAsync(
        string query, List<EntityData> entities, List<RelationData> relations, List<ChunkData> vectorChunks, QueryParam param, CancellationToken ct)
    {
        float[]? queryEmbedding = null;
        if (string.Equals(_options.KgChunkPickMethod, "VECTOR", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                var embedded = await _chunksVdb.EmbeddingFunc.InvokeAsync([query], "query", ct).ConfigureAwait(false);
                queryEmbedding = embedded.Length > 0 ? embedded[0] : null;
            }
            catch
            {
                queryEmbedding = null; // fall back to weighted polling
            }
        }

        var entityChunks = entities.Count > 0
            ? await FindRelatedChunksAsync(entities.Select(e => (e.Name, e.SourceId)).ToList(), null, query, queryEmbedding, "entity", ct).ConfigureAwait(false)
            : [];

        var entityChunkIds = entityChunks.Select(c => c.ChunkId).ToHashSet();
        var relationChunks = relations.Count > 0
            ? await FindRelatedChunksAsync(relations.Select(r => ($"{SortedPair(r.Src, r.Tgt)}", r.SourceId)).ToList(), entityChunkIds, query, queryEmbedding, "relationship", ct).ConfigureAwait(false)
            : [];

        // Round-robin merge: vector, entity, relation; dedup by chunk id.
        var merged = new List<ChunkData>();
        var seen = new HashSet<string>();
        var maxLen = Math.Max(vectorChunks.Count, Math.Max(entityChunks.Count, relationChunks.Count));
        for (var i = 0; i < maxLen; i++)
        {
            AddChunk(vectorChunks, i);
            AddChunk(entityChunks, i);
            AddChunk(relationChunks, i);
        }
        return merged;

        void AddChunk(List<ChunkData> source, int i)
        {
            if (i < source.Count && !string.IsNullOrEmpty(source[i].ChunkId) && seen.Add(source[i].ChunkId))
            {
                merged.Add(source[i]);
            }
        }
    }

    // Ports _find_related_text_unit_from_{entities,relations}: occurrence count -> sort -> pick (VECTOR/WEIGHT).
    private async Task<List<ChunkData>> FindRelatedChunksAsync(
        List<(string Key, string? SourceId)> owners, HashSet<string>? excludeChunkIds, string query, float[]? queryEmbedding, string sourceType, CancellationToken ct)
    {
        // Collect chunk lists per owner.
        var ownerChunks = new List<List<string>>();
        foreach (var (_, sourceId) in owners)
        {
            var chunks = SplitSep(sourceId).ToList();
            if (chunks.Count > 0)
            {
                ownerChunks.Add(chunks);
            }
        }
        if (ownerChunks.Count == 0)
        {
            return [];
        }

        // Count occurrences and dedup (keep first occurrence across owners); optionally exclude entity chunks.
        var occurrence = new Dictionary<string, int>();
        var dedupedOwners = new List<List<string>>();
        foreach (var chunks in ownerChunks)
        {
            var deduped = new List<string>();
            foreach (var id in chunks)
            {
                if (excludeChunkIds is not null && excludeChunkIds.Contains(id))
                {
                    continue;
                }
                occurrence[id] = occurrence.GetValueOrDefault(id) + 1;
                if (occurrence[id] == 1)
                {
                    deduped.Add(id);
                }
            }
            if (deduped.Count > 0)
            {
                dedupedOwners.Add(deduped);
            }
        }
        if (dedupedOwners.Count == 0)
        {
            return [];
        }

        // Sort each owner's chunks by occurrence count (desc).
        var sortedOwners = dedupedOwners
            .Select(chunks => chunks.OrderByDescending(id => occurrence.GetValueOrDefault(id)).ToList())
            .ToList();

        // Pick chunk ids.
        List<string> selected = [];
        if (string.Equals(_options.KgChunkPickMethod, "VECTOR", StringComparison.OrdinalIgnoreCase) && queryEmbedding is not null)
        {
            var numOfChunks = (int)(_options.RelatedChunkNumber * sortedOwners.Count / 2.0);
            selected = await PickByVectorSimilarityAsync(queryEmbedding, sortedOwners, numOfChunks, ct).ConfigureAwait(false);
        }
        if (selected.Count == 0)
        {
            selected = PickByWeightedPolling(sortedOwners, _options.RelatedChunkNumber, 1);
        }
        if (selected.Count == 0)
        {
            return [];
        }

        // Dedup preserving order, then fetch content.
        var uniqueIds = new List<string>();
        var seen = new HashSet<string>();
        foreach (var id in selected)
        {
            if (seen.Add(id))
            {
                uniqueIds.Add(id);
            }
        }

        var records = await _textChunks.GetByIdsAsync(uniqueIds, ct).ConfigureAwait(false);
        var result = new List<ChunkData>();
        for (var i = 0; i < uniqueIds.Count; i++)
        {
            var content = records[i]?.GetString("content");
            if (content is not null)
            {
                result.Add(new ChunkData(content, records[i]!.GetString("file_path") ?? "unknown_source", uniqueIds[i]));
            }
        }
        return result;
    }

    // Linear-gradient weighted polling (port of pick_by_weighted_polling).
    private static List<string> PickByWeightedPolling(List<List<string>> owners, int maxRelatedChunks, int minRelatedChunks)
    {
        if (owners.Count == 0)
        {
            return [];
        }
        if (owners.Count == 1)
        {
            return owners[0].Take(maxRelatedChunks).ToList();
        }

        var n = owners.Count;
        var expected = new int[n];
        for (var i = 0; i < n; i++)
        {
            var ratio = (double)i / (n - 1);
            expected[i] = (int)Math.Round(maxRelatedChunks - ratio * (maxRelatedChunks - minRelatedChunks), MidpointRounding.AwayFromZero);
        }

        var selected = new List<string>();
        var used = new int[n];
        var totalRemaining = 0;
        for (var i = 0; i < n; i++)
        {
            var actual = Math.Min(expected[i], owners[i].Count);
            selected.AddRange(owners[i].Take(actual));
            used[i] = actual;
            if (expected[i] - actual > 0)
            {
                totalRemaining += expected[i] - actual;
            }
        }

        for (var round = 0; round < totalRemaining; round++)
        {
            var allocated = false;
            for (var i = 0; i < n; i++)
            {
                if (used[i] < owners[i].Count)
                {
                    selected.Add(owners[i][used[i]]);
                    used[i]++;
                    allocated = true;
                    break;
                }
            }
            if (!allocated)
            {
                break;
            }
        }
        return selected;
    }

    // Vector-similarity selection (port of pick_by_vector_similarity).
    private async Task<List<string>> PickByVectorSimilarityAsync(float[] queryEmbedding, List<List<string>> owners, int numOfChunks, CancellationToken ct)
    {
        if (numOfChunks <= 0)
        {
            return [];
        }
        var allIds = owners.SelectMany(o => o).Distinct().ToList();
        if (allIds.Count == 0)
        {
            return [];
        }

        IReadOnlyDictionary<string, float[]> vectors;
        try
        {
            vectors = await _chunksVdb.GetVectorsByIdsAsync(allIds, ct).ConfigureAwait(false);
        }
        catch
        {
            return [];
        }
        if (vectors.Count == 0 || vectors.Count != allIds.Count)
        {
            return [];
        }

        var queryNorm = VectorMath.Norm(queryEmbedding);
        var scored = new List<(string Id, double Sim)>();
        foreach (var id in allIds)
        {
            if (vectors.TryGetValue(id, out var vec))
            {
                scored.Add((id, VectorMath.CosineSimilarity(queryEmbedding, vec, queryNorm)));
            }
        }
        return scored.OrderByDescending(s => s.Sim).Take(numOfChunks).Select(s => s.Id).ToList();
    }

    // ---- Stage 4: unified chunk processing (rerank -> chunk_top_k -> token truncate) ----

    private async Task<List<ChunkData>> ProcessChunksUnifiedAsync(string query, List<ChunkData> chunks, QueryParam param, int chunkTokenLimit, CancellationToken ct)
    {
        if (chunks.Count == 0)
        {
            return [];
        }

        // 1+2. Rerank and min-score filter (only when a reranker is configured; otherwise a no-op as in Python).
        if (param.EnableRerank && _options.Reranker is not null)
        {
            var topN = param.ChunkTopK > 0 ? param.ChunkTopK : chunks.Count;
            var scores = await _options.Reranker(query, chunks.Select(c => c.Content).ToList(), ct).ConfigureAwait(false);
            chunks = chunks.Zip(scores, (c, s) => (Chunk: c, Score: s))
                .OrderByDescending(x => x.Score)
                .Take(topN)
                .Where(x => x.Score >= _options.MinRerankScore)
                .Select(x => x.Chunk)
                .ToList();
            if (chunks.Count == 0)
            {
                return [];
            }
        }

        // 3. chunk_top_k cap.
        if (param.ChunkTopK > 0 && chunks.Count > param.ChunkTopK)
        {
            chunks = chunks.Take(param.ChunkTopK).ToList();
        }

        // 4. Token-based truncation against the dynamic budget.
        chunks = TextUtils.TruncateListByTokenSize(chunks,
            c => JsonHelper.SerializeLine(("content", c.Content), ("file_path", c.FilePath), ("chunk_id", c.ChunkId)),
            chunkTokenLimit, _tokenizer).ToList();

        return chunks;
    }

    // ---- reference list + serialization ----

    private static (List<ReferenceEntry> RefList, Dictionary<string, string> FileToRef) GenerateReferenceList(List<ChunkData> chunks)
    {
        var refList = new List<ReferenceEntry>();
        var fileToRef = new Dictionary<string, string>();
        if (chunks.Count == 0)
        {
            return (refList, fileToRef);
        }

        // Count occurrences and first-appearance order, excluding unknown_source.
        var counts = new Dictionary<string, int>();
        var firstIndex = new Dictionary<string, int>();
        for (var i = 0; i < chunks.Count; i++)
        {
            var fp = chunks[i].FilePath;
            if (string.IsNullOrEmpty(fp) || fp == "unknown_source")
            {
                continue;
            }
            counts[fp] = counts.GetValueOrDefault(fp) + 1;
            if (!firstIndex.ContainsKey(fp))
            {
                firstIndex[fp] = i;
            }
        }

        // Sort by count desc, then first-appearance asc.
        var ordered = firstIndex.Keys
            .OrderByDescending(fp => counts[fp])
            .ThenBy(fp => firstIndex[fp])
            .ToList();
        for (var i = 0; i < ordered.Count; i++)
        {
            var refId = (i + 1).ToString();
            fileToRef[ordered[i]] = refId;
            refList.Add(new ReferenceEntry(refId, ordered[i]));
        }
        return (refList, fileToRef);
    }

    private (string ChunksStr, string ReferenceStr) SerializeChunks(List<ChunkData> chunks, Dictionary<string, string> fileToRef, List<ReferenceEntry> refList)
    {
        var lines = chunks.Select(c =>
        {
            var refId = (!string.IsNullOrEmpty(c.FilePath) && c.FilePath != "unknown_source" && fileToRef.TryGetValue(c.FilePath, out var r)) ? r : "";
            return JsonHelper.SerializeLine(("reference_id", (object?)refId), ("content", c.Content));
        });
        var referenceStr = string.Join("\n", refList.Where(r => !string.IsNullOrEmpty(r.ReferenceId)).Select(r => $"[{r.ReferenceId}] {r.FilePath}"));
        return (string.Join("\n", lines), referenceStr);
    }

    private static IReadOnlyDictionary<string, object?> BuildRawData(
        List<EntityContext> entities, List<RelationContext> relations, List<ChunkData> chunks, List<ReferenceEntry> refList)
    {
        return new Dictionary<string, object?>
        {
            ["data"] = new Dictionary<string, object?>
            {
                ["entities"] = entities.Select(e => (object?)new Dictionary<string, object?> { ["entity"] = e.Name, ["type"] = e.Type, ["description"] = e.Description }).ToList(),
                ["relationships"] = relations.Select(r => (object?)new Dictionary<string, object?> { ["src"] = r.Entity1, ["tgt"] = r.Entity2, ["description"] = r.Description }).ToList(),
                ["chunks"] = chunks.Select(c => (object?)new Dictionary<string, object?> { ["content"] = c.Content, ["file_path"] = c.FilePath }).ToList(),
                ["references"] = refList.Select(r => (object?)new Dictionary<string, object?> { ["reference_id"] = r.ReferenceId, ["file_path"] = r.FilePath }).ToList(),
            },
        };
    }

    // ---- helpers ----

    private static (string, string) SortedPair(string a, string b)
        => string.CompareOrdinal(a, b) <= 0 ? (a, b) : (b, a);

    private static double GetWeight(StorageRecord edge)
        => edge.TryGetValue("weight", out var w) && w is not null ? Convert.ToDouble(w) : 1.0;

    private static long? GetCreatedAt(StorageRecord? r)
    {
        if (r is null || !r.TryGetValue("created_at", out var v) || v is null)
        {
            return null;
        }
        return v switch
        {
            long l => l,
            int i => i,
            double d => (long)d,
            string s when long.TryParse(s, out var p) => p,
            _ => null,
        };
    }

    private static string FormatCreatedAt(long? createdAt)
        => createdAt is null
            ? "UNKNOWN"
            : DateTimeOffset.FromUnixTimeSeconds(createdAt.Value).ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss");

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

    private sealed record EntityData(string Name, string Type, string Description, string FilePath, string? SourceId, long? CreatedAt, int Rank);
    private sealed record RelationData(string Src, string Tgt, string Keywords, string Description, string FilePath, string? SourceId, long? CreatedAt, double Weight, int Rank);
    private sealed record ChunkData(string Content, string FilePath, string ChunkId);
    private sealed record ReferenceEntry(string ReferenceId, string FilePath);
}
