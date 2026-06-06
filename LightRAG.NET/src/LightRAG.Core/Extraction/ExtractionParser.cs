using LightRAG.Core.Configuration;
using LightRAG.Core.Prompts;
using LightRAG.Core.Utils;
using Newtonsoft.Json.Linq;

namespace LightRAG.Core.Extraction;

/// <summary>
/// Parses LLM extraction output into nodes/edges. Ports <c>_process_extraction_result</c> (text mode),
/// <c>_process_json_extraction_result</c> (JSON mode), and the <c>_handle_single_*_extraction</c> validators
/// from <c>lightrag/operate.py</c>.
/// </summary>
public static class ExtractionParser
{
    private static readonly char[] InvalidEntityTypeChars = ['\'', '(', ')', '<', '>', '|', '/', '\\'];

    public static ChunkExtractionResult ParseTextMode(
        string result,
        string chunkKey,
        long timestamp,
        string filePath,
        string tupleDelimiter = PromptTemplates.DefaultTupleDelimiter,
        string completionDelimiter = PromptTemplates.DefaultCompletionDelimiter)
    {
        var output = new ChunkExtractionResult();

        var records = TextUtils.SplitStringByMultiMarkers(
            result,
            ["\n", completionDelimiter, completionDelimiter.ToLowerInvariant()]);

        foreach (var raw in records)
        {
            var record = raw.Trim();
            if (record.Length == 0)
            {
                continue;
            }

            var attrs = TextUtils.SplitStringByMultiMarkers(record, [tupleDelimiter]);
            attrs = NormalizeMisPrefixedRelation(attrs);

            var node = HandleEntity(attrs, chunkKey, timestamp, filePath);
            if (node is not null)
            {
                var name = Truncate(node.EntityName, Constants.DefaultEntityNameMaxLength);
                var fixedNode = node with { EntityName = name };
                AddNode(output, name, fixedNode);
                continue;
            }

            var edge = HandleRelationship(attrs, chunkKey, timestamp, filePath);
            if (edge is not null)
            {
                var src = Truncate(edge.SrcId, Constants.DefaultEntityNameMaxLength);
                var tgt = Truncate(edge.TgtId, Constants.DefaultEntityNameMaxLength);
                var fixedEdge = edge with { SrcId = src, TgtId = tgt };
                AddEdge(output, (src, tgt), fixedEdge);
            }
        }

        return output;
    }

    public static ChunkExtractionResult ParseJsonMode(
        string result,
        string chunkKey,
        long timestamp,
        string filePath)
    {
        var output = new ChunkExtractionResult();
        var token = JsonRepair.TryParse(result);
        if (token is not JObject root)
        {
            return output;
        }

        if (root["entities"] is JArray entities)
        {
            foreach (var e in entities)
            {
                var name = TextUtils.SanitizeAndNormalize(GetStr(e, "name"), removeInnerQuotes: true);
                var type = TextUtils.SanitizeAndNormalize(GetStr(e, "type"), removeInnerQuotes: true);
                var description = TextUtils.SanitizeAndNormalize(GetStr(e, "description"));
                if (name.Length == 0 || description.Length == 0)
                {
                    continue;
                }
                type = NormalizeEntityType(type);
                if (type is null)
                {
                    continue;
                }
                name = Truncate(name, Constants.DefaultEntityNameMaxLength);
                AddNode(output, name, new ExtractedNode
                {
                    EntityName = name,
                    EntityType = type,
                    Description = description,
                    SourceId = chunkKey,
                    FilePath = filePath,
                    Timestamp = timestamp,
                });
            }
        }

        if (root["relationships"] is JArray rels)
        {
            foreach (var r in rels)
            {
                var src = TextUtils.SanitizeAndNormalize(GetStr(r, "source"), removeInnerQuotes: true);
                var tgt = TextUtils.SanitizeAndNormalize(GetStr(r, "target"), removeInnerQuotes: true);
                var description = TextUtils.SanitizeAndNormalize(GetStr(r, "description"));
                var keywords = TextUtils.SanitizeAndNormalize(GetStr(r, "keywords"), removeInnerQuotes: true);
                if (src.Length == 0 || tgt.Length == 0 || src == tgt || description.Length == 0)
                {
                    continue;
                }
                src = Truncate(src, Constants.DefaultEntityNameMaxLength);
                tgt = Truncate(tgt, Constants.DefaultEntityNameMaxLength);
                AddEdge(output, (src, tgt), new ExtractedEdge
                {
                    SrcId = src,
                    TgtId = tgt,
                    Weight = 1.0,
                    Description = description,
                    Keywords = keywords,
                    SourceId = chunkKey,
                    FilePath = filePath,
                    Timestamp = timestamp,
                });
            }
        }

        return output;
    }

    // ---- record validators (ported from _handle_single_*_extraction) ----

    private static ExtractedNode? HandleEntity(List<string> attrs, string chunkKey, long timestamp, string filePath)
    {
        if (attrs.Count != 4 || !attrs[0].Contains("entity"))
        {
            return null;
        }

        var name = TextUtils.SanitizeAndNormalize(attrs[1], removeInnerQuotes: true);
        if (string.IsNullOrWhiteSpace(name))
        {
            return null;
        }

        var type = NormalizeEntityType(TextUtils.SanitizeAndNormalize(attrs[2], removeInnerQuotes: true));
        if (type is null)
        {
            return null;
        }

        var description = TextUtils.SanitizeAndNormalize(attrs[3]);
        if (string.IsNullOrWhiteSpace(description))
        {
            return null;
        }

        return new ExtractedNode
        {
            EntityName = name,
            EntityType = type,
            Description = description,
            SourceId = chunkKey,
            FilePath = filePath,
            Timestamp = timestamp,
        };
    }

    private static ExtractedEdge? HandleRelationship(List<string> attrs, string chunkKey, long timestamp, string filePath)
    {
        if (attrs.Count != 5 || !attrs[0].Contains("relation"))
        {
            return null;
        }

        var source = TextUtils.SanitizeAndNormalize(attrs[1], removeInnerQuotes: true);
        var target = TextUtils.SanitizeAndNormalize(attrs[2], removeInnerQuotes: true);
        if (string.IsNullOrEmpty(source) || string.IsNullOrEmpty(target) || source == target)
        {
            return null;
        }

        var keywords = TextUtils.SanitizeAndNormalize(attrs[3], removeInnerQuotes: true).Replace('，', ',');
        var description = TextUtils.SanitizeAndNormalize(attrs[4]);
        if (string.IsNullOrWhiteSpace(description))
        {
            return null;
        }

        var lastField = attrs[^1].Trim('"', '\'');
        var weight = TextUtils.IsFloatRegex(lastField) ? double.Parse(lastField) : 1.0;

        return new ExtractedEdge
        {
            SrcId = source,
            TgtId = target,
            Weight = weight,
            Description = description,
            Keywords = keywords,
            SourceId = chunkKey,
            FilePath = filePath,
            Timestamp = timestamp,
        };
    }

    /// <summary>Lowercase, strip spaces, reject invalid chars, take first comma token. Returns null if invalid.</summary>
    private static string? NormalizeEntityType(string type)
    {
        if (string.IsNullOrWhiteSpace(type) || type.IndexOfAny(InvalidEntityTypeChars) >= 0)
        {
            return null;
        }
        if (type.Contains(','))
        {
            var firstNonEmpty = type.Split(',').Select(t => t.Trim()).FirstOrDefault(t => t.Length > 0);
            if (firstNonEmpty is null)
            {
                return null;
            }
            type = firstNonEmpty;
        }
        return type.Replace(" ", "").ToLowerInvariant();
    }

    /// <summary>Recover the text-mode failure where a relation row uses the "entity" prefix.</summary>
    private static List<string> NormalizeMisPrefixedRelation(List<string> attrs)
    {
        if (attrs.Count != 5)
        {
            return attrs;
        }
        var prefix = attrs[0].Trim().ToLowerInvariant();
        if (!prefix.Contains("entity") || prefix.Contains("relation"))
        {
            return attrs;
        }
        var normalized = new List<string>(attrs) { [0] = "relation" };
        return normalized;
    }

    private static void AddNode(ChunkExtractionResult output, string name, ExtractedNode node)
    {
        if (!output.Nodes.TryGetValue(name, out var list))
        {
            list = [];
            output.Nodes[name] = list;
        }
        list.Add(node);
    }

    private static void AddEdge(ChunkExtractionResult output, (string, string) key, ExtractedEdge edge)
    {
        if (!output.Edges.TryGetValue(key, out var list))
        {
            list = [];
            output.Edges[key] = list;
        }
        list.Add(edge);
    }

    private static string Truncate(string value, int maxLength)
        => value.Length <= maxLength ? value : value[..maxLength];

    private static string GetStr(JToken element, string property)
        => element is JObject obj && obj[property] is { Type: JTokenType.String } p
            ? p.Value<string>() ?? string.Empty
            : string.Empty;
}
