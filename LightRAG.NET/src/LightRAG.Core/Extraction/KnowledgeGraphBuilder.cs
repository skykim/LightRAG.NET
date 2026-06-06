using LightRAG.Core.Abstractions;
using LightRAG.Core.Configuration;
using LightRAG.Core.Llm;
using LightRAG.Core.Prompts;
using LightRAG.Core.Utils;

namespace LightRAG.Core.Extraction;

/// <summary>Configuration for the knowledge-graph merge step.</summary>
public sealed record MergeOptions
{
    public string Language { get; init; } = Constants.DefaultSummaryLanguage;
    public int ForceLlmSummaryOnMerge { get; init; } = Constants.DefaultForceLlmSummaryOnMerge;
    public int SummaryMaxTokens { get; init; } = Constants.DefaultSummaryMaxTokens;
    public int SummaryContextSize { get; init; } = Constants.DefaultSummaryContextSize;
    public int SummaryLengthRecommended { get; init; } = Constants.DefaultSummaryLengthRecommended;
    public int MaxSourceIdsPerEntity { get; init; } = Constants.DefaultMaxSourceIdsPerEntity;
    public int MaxSourceIdsPerRelation { get; init; } = Constants.DefaultMaxSourceIdsPerRelation;
    public int MaxFilePaths { get; init; } = Constants.DefaultMaxFilePaths;
}

/// <summary>
/// Merges per-chunk extraction results into the knowledge graph and entity/relationship vector stores.
/// Ports <c>merge_nodes_and_edges</c>, <c>_merge_nodes_then_upsert</c>, <c>_merge_edges_then_upsert</c>,
/// and <c>_handle_entity_relation_summary</c> from <c>lightrag/operate.py</c>.
///
/// Writes are serialized (entities first, then edges) because the file-based graph/vector stores are
/// not built for concurrent mutation; the heavy cost (LLM summaries) only triggers on re-merges.
/// </summary>
public sealed class KnowledgeGraphBuilder
{
    private const string Sep = Constants.GraphFieldSep;

    private readonly IGraphStorage _graph;
    private readonly IVectorStorage _entityVdb;
    private readonly IVectorStorage _relationshipsVdb;
    private readonly ITokenizer _tokenizer;
    private readonly ILlmCaller _summaryLlm;
    private readonly MergeOptions _options;

    public KnowledgeGraphBuilder(
        IGraphStorage graph,
        IVectorStorage entityVdb,
        IVectorStorage relationshipsVdb,
        ITokenizer tokenizer,
        ILlmCaller summaryLlm,
        MergeOptions? options = null)
    {
        _graph = graph;
        _entityVdb = entityVdb;
        _relationshipsVdb = relationshipsVdb;
        _tokenizer = tokenizer;
        _summaryLlm = summaryLlm;
        _options = options ?? new MergeOptions();
    }

    public async Task MergeAsync(IReadOnlyList<ChunkExtractionResult> chunkResults, CancellationToken cancellationToken = default)
    {
        // Aggregate across chunks. Edges use a canonical (sorted) key for the undirected graph.
        var allNodes = new Dictionary<string, List<ExtractedNode>>();
        var allEdges = new Dictionary<(string, string), List<ExtractedEdge>>();

        foreach (var chunk in chunkResults)
        {
            foreach (var (name, nodes) in chunk.Nodes)
            {
                if (!allNodes.TryGetValue(name, out var list))
                {
                    list = [];
                    allNodes[name] = list;
                }
                list.AddRange(nodes);
            }
            foreach (var (key, edges) in chunk.Edges)
            {
                var sortedKey = string.CompareOrdinal(key.Item1, key.Item2) <= 0 ? key : (key.Item2, key.Item1);
                if (!allEdges.TryGetValue(sortedKey, out var list))
                {
                    list = [];
                    allEdges[sortedKey] = list;
                }
                list.AddRange(edges);
            }
        }

        // Phase 1: entities (must precede edges so edge endpoints resolve).
        foreach (var (name, nodes) in allNodes)
        {
            await MergeNodeAsync(name, nodes, cancellationToken).ConfigureAwait(false);
        }

        // Phase 2: relationships (may create missing endpoint entities).
        foreach (var (key, edges) in allEdges)
        {
            await MergeEdgeAsync(key.Item1, key.Item2, edges, cancellationToken).ConfigureAwait(false);
        }
    }

    private async Task MergeNodeAsync(string entityName, List<ExtractedNode> nodesData, CancellationToken cancellationToken)
    {
        var alreadyTypes = new List<string>();
        var alreadySourceIds = new List<string>();
        var alreadyDescriptions = new List<string>();
        var alreadyFilePaths = new List<string>();

        var alreadyNode = await _graph.GetNodeAsync(entityName, cancellationToken).ConfigureAwait(false);
        if (alreadyNode is not null)
        {
            var existingType = alreadyNode.GetString("entity_type");
            existingType = string.IsNullOrWhiteSpace(existingType) ? "UNKNOWN" : existingType!;
            if (existingType.Contains(','))
            {
                existingType = existingType.Split(',').Select(t => t.Trim()).FirstOrDefault(t => t.Length > 0) ?? "UNKNOWN";
            }
            alreadyTypes.Add(existingType);
            alreadySourceIds.AddRange(SplitSep(alreadyNode.GetString("source_id")));
            alreadyFilePaths.AddRange(SplitSep(alreadyNode.GetString("file_path") ?? "unknown_source"));
            alreadyDescriptions.AddRange(SplitSep(alreadyNode.GetString("description")));
        }

        var newSourceIds = nodesData.Select(n => n.SourceId).Where(s => !string.IsNullOrEmpty(s));
        var sourceIds = CapKeep(MergeOrdered(alreadySourceIds, newSourceIds), _options.MaxSourceIdsPerEntity);

        var entityType = MostCommon(nodesData.Select(n => n.EntityType).Concat(alreadyTypes)) ?? "UNKNOWN";

        // Dedup by description, sort by (timestamp asc, length desc).
        var uniqueByDesc = new Dictionary<string, ExtractedNode>();
        foreach (var node in nodesData)
        {
            if (!string.IsNullOrEmpty(node.Description) && !uniqueByDesc.ContainsKey(node.Description))
            {
                uniqueByDesc[node.Description] = node;
            }
        }
        var sortedNew = uniqueByDesc.Values
            .OrderBy(n => n.Timestamp).ThenByDescending(n => n.Description.Length)
            .Select(n => n.Description);
        var descriptionList = alreadyDescriptions.Concat(sortedNew).ToList();
        if (descriptionList.Count == 0)
        {
            descriptionList.Add($"Entity {entityName}");
        }

        var (description, _) = await SummarizeAsync("Entity", entityName, descriptionList, cancellationToken).ConfigureAwait(false);
        var filePaths = CapKeep(MergeOrdered(alreadyFilePaths.Where(fp => !fp.StartsWith("...")), nodesData.Select(n => n.FilePath)), _options.MaxFilePaths);

        var sourceId = string.Join(Sep, sourceIds);
        var filePath = string.Join(Sep, filePaths);

        var nodeData = new Dictionary<string, object?>
        {
            ["entity_id"] = entityName,
            ["entity_type"] = entityType,
            ["description"] = description,
            ["source_id"] = sourceId,
            ["file_path"] = filePath,
            ["created_at"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
        };
        await _graph.UpsertNodeAsync(entityName, nodeData, cancellationToken).ConfigureAwait(false);

        var entityVdbId = Hashing.ComputeMdHashId(entityName, "ent-");
        await _entityVdb.UpsertAsync(new Dictionary<string, StorageRecord>
        {
            [entityVdbId] = new()
            {
                ["entity_name"] = entityName,
                ["entity_type"] = entityType,
                ["content"] = $"{entityName}\n{description}",
                ["source_id"] = sourceId,
                ["file_path"] = filePath,
            },
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task MergeEdgeAsync(string srcId, string tgtId, List<ExtractedEdge> edgesData, CancellationToken cancellationToken)
    {
        if (srcId == tgtId)
        {
            return;
        }

        var alreadyWeights = new List<double>();
        var alreadySourceIds = new List<string>();
        var alreadyDescriptions = new List<string>();
        var alreadyKeywords = new List<string>();
        var alreadyFilePaths = new List<string>();

        var alreadyEdge = await _graph.GetEdgeAsync(srcId, tgtId, cancellationToken).ConfigureAwait(false);
        if (alreadyEdge is not null)
        {
            if (alreadyEdge.TryGetValue("weight", out var w) && w is not null)
            {
                alreadyWeights.Add(Convert.ToDouble(w));
            }
            alreadySourceIds.AddRange(SplitSep(alreadyEdge.GetString("source_id")));
            alreadyFilePaths.AddRange(SplitSep(alreadyEdge.GetString("file_path")));
            alreadyDescriptions.AddRange(SplitSep(alreadyEdge.GetString("description")));
            alreadyKeywords.AddRange(SplitSep(alreadyEdge.GetString("keywords")));
        }

        var weight = edgesData.Sum(e => e.Weight) + alreadyWeights.Sum();

        var keywordSet = new SortedSet<string>(StringComparer.Ordinal);
        foreach (var k in alreadyKeywords.Concat(edgesData.Select(e => e.Keywords)))
        {
            foreach (var part in (k ?? string.Empty).Split(','))
            {
                var trimmed = part.Trim();
                if (trimmed.Length > 0)
                {
                    keywordSet.Add(trimmed);
                }
            }
        }
        var keywords = string.Join(",", keywordSet);

        var newSourceIds = edgesData.Select(e => e.SourceId).Where(s => !string.IsNullOrEmpty(s));
        var sourceIds = CapKeep(MergeOrdered(alreadySourceIds, newSourceIds), _options.MaxSourceIdsPerRelation);

        var uniqueByDesc = new Dictionary<string, ExtractedEdge>();
        foreach (var edge in edgesData)
        {
            if (!string.IsNullOrEmpty(edge.Description) && !uniqueByDesc.ContainsKey(edge.Description))
            {
                uniqueByDesc[edge.Description] = edge;
            }
        }
        var sortedNew = uniqueByDesc.Values
            .OrderBy(e => e.Timestamp).ThenByDescending(e => e.Description.Length)
            .Select(e => e.Description);
        var descriptionList = alreadyDescriptions.Concat(sortedNew).ToList();
        if (descriptionList.Count == 0)
        {
            descriptionList.Add($"{srcId} is related to {tgtId}");
        }

        var (description, _) = await SummarizeAsync("Relation", $"({srcId}, {tgtId})", descriptionList, cancellationToken).ConfigureAwait(false);
        var filePaths = CapKeep(MergeOrdered(alreadyFilePaths.Where(fp => !fp.StartsWith("...")), edgesData.Select(e => e.FilePath)), _options.MaxFilePaths);

        var sourceId = string.Join(Sep, sourceIds);
        var filePath = string.Join(Sep, filePaths);

        // Ensure both endpoints exist (create UNKNOWN-type entities if missing).
        foreach (var endpoint in new[] { srcId, tgtId })
        {
            var existing = await _graph.GetNodeAsync(endpoint, cancellationToken).ConfigureAwait(false);
            if (existing is null)
            {
                await _graph.UpsertNodeAsync(endpoint, new Dictionary<string, object?>
                {
                    ["entity_id"] = endpoint,
                    ["source_id"] = sourceId,
                    ["description"] = description,
                    ["entity_type"] = "UNKNOWN",
                    ["file_path"] = filePath,
                    ["created_at"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                }, cancellationToken).ConfigureAwait(false);

                var endpointVdbId = Hashing.ComputeMdHashId(endpoint, "ent-");
                await _entityVdb.UpsertAsync(new Dictionary<string, StorageRecord>
                {
                    [endpointVdbId] = new()
                    {
                        ["entity_name"] = endpoint,
                        ["entity_type"] = "UNKNOWN",
                        ["content"] = $"{endpoint}\n{description}",
                        ["source_id"] = sourceId,
                        ["file_path"] = filePath,
                    },
                }, cancellationToken).ConfigureAwait(false);
            }
        }

        await _graph.UpsertEdgeAsync(srcId, tgtId, new Dictionary<string, object?>
        {
            ["weight"] = weight,
            ["description"] = description,
            ["keywords"] = keywords,
            ["source_id"] = sourceId,
            ["file_path"] = filePath,
            ["created_at"] = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
        }, cancellationToken).ConfigureAwait(false);

        // Relationship VDB: canonical ordering, id = rel-md5(src+tgt).
        var (a, b) = string.CompareOrdinal(srcId, tgtId) <= 0 ? (srcId, tgtId) : (tgtId, srcId);
        var relVdbId = Hashing.ComputeMdHashId(a + b, "rel-");
        var relVdbIdReverse = Hashing.ComputeMdHashId(b + a, "rel-");
        await _relationshipsVdb.DeleteAsync([relVdbId, relVdbIdReverse], cancellationToken).ConfigureAwait(false);
        await _relationshipsVdb.UpsertAsync(new Dictionary<string, StorageRecord>
        {
            [relVdbId] = new()
            {
                ["src_id"] = a,
                ["tgt_id"] = b,
                ["source_id"] = sourceId,
                ["content"] = $"{keywords}\t{a}\n{b}\n{description}",
                ["keywords"] = keywords,
                ["description"] = description,
                ["weight"] = weight,
                ["file_path"] = filePath,
            },
        }, cancellationToken).ConfigureAwait(false);
    }

    private async Task<(string Description, bool LlmUsed)> SummarizeAsync(
        string type, string name, List<string> descriptions, CancellationToken cancellationToken)
    {
        if (descriptions.Count == 0)
        {
            return (string.Empty, false);
        }
        if (descriptions.Count == 1)
        {
            return (descriptions[0], false);
        }

        var totalTokens = descriptions.Sum(d => _tokenizer.CountTokens(d));
        if (descriptions.Count < _options.ForceLlmSummaryOnMerge && totalTokens < _options.SummaryMaxTokens)
        {
            return (string.Join(Sep, descriptions), false);
        }

        var summary = await SummarizeWithLlmAsync(type, name, descriptions, cancellationToken).ConfigureAwait(false);
        return (string.IsNullOrWhiteSpace(summary) ? string.Join(Sep, descriptions) : summary, true);
    }

    private async Task<string> SummarizeWithLlmAsync(string type, string name, List<string> descriptions, CancellationToken cancellationToken)
    {
        // Truncate the description list to the summary context budget, JSONL-encoded as {"Description": ...}.
        var jsonl = TextUtils.TruncateListByTokenSize(
            descriptions,
            d => JsonHelper.Serialize(new { Description = d }, indented: false),
            _options.SummaryContextSize,
            _tokenizer);

        var joined = string.Join("\n", jsonl.Select(d => JsonHelper.Serialize(new { Description = d }, indented: false)));

        var prompt = PromptRenderer.Render(PromptTemplates.SummarizeEntityDescriptions,
            ("description_type", type),
            ("description_name", name),
            ("description_list", joined),
            ("summary_length", _options.SummaryLengthRecommended.ToString()),
            ("language", _options.Language));

        return await _summaryLlm.CompleteAsync(prompt, cacheType: "summary", cancellationToken: cancellationToken).ConfigureAwait(false);
    }

    // ---- helpers ----

    private static IEnumerable<string> SplitSep(string? value)
        => string.IsNullOrEmpty(value) ? [] : value!.Split(new[] { Sep }, StringSplitOptions.RemoveEmptyEntries);

    private static List<string> MergeOrdered(IEnumerable<string> existing, IEnumerable<string> incoming)
    {
        var result = new List<string>();
        var seen = new HashSet<string>();
        foreach (var id in existing.Concat(incoming))
        {
            if (!string.IsNullOrEmpty(id) && seen.Add(id))
            {
                result.Add(id);
            }
        }
        return result;
    }

    private static List<string> CapKeep(List<string> items, int max)
        => items.Count <= max ? items : items.Take(max).ToList();

    private static string? MostCommon(IEnumerable<string> values)
    {
        var counts = new Dictionary<string, int>();
        var order = new Dictionary<string, int>();
        var i = 0;
        foreach (var v in values)
        {
            if (string.IsNullOrEmpty(v))
            {
                continue;
            }
            counts[v] = counts.GetValueOrDefault(v) + 1;
            if (!order.ContainsKey(v))
            {
                order[v] = i++;
            }
        }
        return counts.Count == 0
            ? null
            : counts.OrderByDescending(kv => kv.Value).ThenBy(kv => order[kv.Key]).First().Key;
    }
}
