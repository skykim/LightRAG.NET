using System.Globalization;
using System.Xml.Linq;
using LightRAG.Core.Abstractions;

namespace LightRAG.Storage.FileBased;

/// <summary>
/// In-memory undirected knowledge graph with GraphML persistence, a clean reimplementation of
/// <c>NetworkXStorage</c> (<c>lightrag/kg/networkx_impl.py</c>). Persists to
/// <c>graph_{namespace}.graphml</c>. All edges are undirected (stored under a canonical key pair).
/// </summary>
public sealed class GraphmlGraphStorage : FileStorageBase, IGraphStorage
{
    private static readonly XNamespace Gml = "http://graphml.graphdrawing.org/xmlns";

    private readonly Dictionary<string, Dictionary<string, object?>> _nodes = new();
    private readonly Dictionary<(string, string), Dictionary<string, object?>> _edges = new();
    private readonly Dictionary<string, HashSet<string>> _adjacency = new();
    private readonly string _filePath;
    private bool _dirty;

    public EmbeddingFunc EmbeddingFunc { get; }

    public GraphmlGraphStorage(
        string workingDir,
        EmbeddingFunc embeddingFunc,
        string @namespace = NameSpace.GraphStoreChunkEntityRelation,
        string workspace = "")
        : base(workingDir, @namespace, workspace)
    {
        EmbeddingFunc = embeddingFunc;
        _filePath = Path.Combine(Directory, $"graph_{@namespace}.graphml");
    }

    private static (string, string) Canon(string a, string b)
        => string.CompareOrdinal(a, b) <= 0 ? (a, b) : (b, a);

    public Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        EnsureDirectory();
        SweepOrphanTmp(_filePath);
        LoadFromDisk();
        return Task.CompletedTask;
    }

    public Task<bool> HasNodeAsync(string nodeId, CancellationToken cancellationToken = default)
        => Task.FromResult(_nodes.ContainsKey(nodeId));

    public Task<bool> HasEdgeAsync(string sourceNodeId, string targetNodeId, CancellationToken cancellationToken = default)
        => Task.FromResult(_edges.ContainsKey(Canon(sourceNodeId, targetNodeId)));

    public Task<int> NodeDegreeAsync(string nodeId, CancellationToken cancellationToken = default)
        => Task.FromResult(_adjacency.TryGetValue(nodeId, out var neighbors) ? neighbors.Count : 0);

    public async Task<int> EdgeDegreeAsync(string srcId, string tgtId, CancellationToken cancellationToken = default)
        => await NodeDegreeAsync(srcId, cancellationToken).ConfigureAwait(false)
         + await NodeDegreeAsync(tgtId, cancellationToken).ConfigureAwait(false);

    public Task<StorageRecord?> GetNodeAsync(string nodeId, CancellationToken cancellationToken = default)
        => Task.FromResult(_nodes.TryGetValue(nodeId, out var data) ? new StorageRecord(data) : null);

    public Task<StorageRecord?> GetEdgeAsync(string sourceNodeId, string targetNodeId, CancellationToken cancellationToken = default)
        => Task.FromResult(_edges.TryGetValue(Canon(sourceNodeId, targetNodeId), out var data) ? new StorageRecord(data) : null);

    public Task<IReadOnlyList<(string Source, string Target)>?> GetNodeEdgesAsync(string sourceNodeId, CancellationToken cancellationToken = default)
    {
        if (!_adjacency.TryGetValue(sourceNodeId, out var neighbors))
        {
            return Task.FromResult<IReadOnlyList<(string, string)>?>(null);
        }
        IReadOnlyList<(string, string)> edges = neighbors.Select(n => (sourceNodeId, n)).ToList();
        return Task.FromResult<IReadOnlyList<(string, string)>?>(edges);
    }

    public Task UpsertNodeAsync(string nodeId, IReadOnlyDictionary<string, object?> nodeData, CancellationToken cancellationToken = default)
    {
        _nodes[nodeId] = new Dictionary<string, object?>(nodeData);
        _adjacency.TryAdd(nodeId, []);
        _dirty = true;
        return Task.CompletedTask;
    }

    public Task UpsertEdgeAsync(string sourceNodeId, string targetNodeId, IReadOnlyDictionary<string, object?> edgeData, CancellationToken cancellationToken = default)
    {
        // Ensure endpoints exist so degree/adjacency stay consistent.
        _nodes.TryAdd(sourceNodeId, []);
        _nodes.TryAdd(targetNodeId, []);
        _adjacency.TryAdd(sourceNodeId, []);
        _adjacency.TryAdd(targetNodeId, []);

        _edges[Canon(sourceNodeId, targetNodeId)] = new Dictionary<string, object?>(edgeData);
        _adjacency[sourceNodeId].Add(targetNodeId);
        _adjacency[targetNodeId].Add(sourceNodeId);
        _dirty = true;
        return Task.CompletedTask;
    }

    public Task DeleteNodeAsync(string nodeId, CancellationToken cancellationToken = default)
    {
        if (_nodes.Remove(nodeId))
        {
            _dirty = true;
        }
        if (_adjacency.TryGetValue(nodeId, out var neighbors))
        {
            foreach (var neighbor in neighbors)
            {
                _adjacency[neighbor].Remove(nodeId);
                _edges.Remove(Canon(nodeId, neighbor));
            }
            _adjacency.Remove(nodeId);
        }
        return Task.CompletedTask;
    }

    public async Task RemoveNodesAsync(IReadOnlyList<string> nodes, CancellationToken cancellationToken = default)
    {
        foreach (var node in nodes)
        {
            await DeleteNodeAsync(node, cancellationToken).ConfigureAwait(false);
        }
    }

    public Task RemoveEdgesAsync(IReadOnlyList<(string Source, string Target)> edges, CancellationToken cancellationToken = default)
    {
        foreach (var (source, target) in edges)
        {
            if (_edges.Remove(Canon(source, target)))
            {
                _adjacency.GetValueOrDefault(source)?.Remove(target);
                _adjacency.GetValueOrDefault(target)?.Remove(source);
                _dirty = true;
            }
        }
        return Task.CompletedTask;
    }

    public Task<IReadOnlyList<string>> GetAllLabelsAsync(CancellationToken cancellationToken = default)
    {
        IReadOnlyList<string> labels = _nodes.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();
        return Task.FromResult(labels);
    }

    public Task<IReadOnlyList<StorageRecord>> GetAllNodesAsync(CancellationToken cancellationToken = default)
    {
        IReadOnlyList<StorageRecord> result = _nodes.Select(kv =>
        {
            var record = new StorageRecord(kv.Value) { ["id"] = kv.Key };
            return record;
        }).ToList();
        return Task.FromResult(result);
    }

    public Task<IReadOnlyList<StorageRecord>> GetAllEdgesAsync(CancellationToken cancellationToken = default)
    {
        IReadOnlyList<StorageRecord> result = _edges.Select(kv =>
        {
            var record = new StorageRecord(kv.Value) { ["source"] = kv.Key.Item1, ["target"] = kv.Key.Item2 };
            return record;
        }).ToList();
        return Task.FromResult(result);
    }

    public Task<KnowledgeGraph> GetKnowledgeGraphAsync(string nodeLabel, int maxDepth = 3, int maxNodes = 1000, CancellationToken cancellationToken = default)
    {
        int Degree(string n) => _adjacency.TryGetValue(n, out var nb) ? nb.Count : 0;

        List<string> selected;
        var truncated = false;

        if (nodeLabel == "*")
        {
            // Highest-degree nodes first, then cap to maxNodes (ports networkx get_knowledge_graph "*").
            var sorted = _nodes.Keys.OrderByDescending(Degree).ToList();
            truncated = sorted.Count > maxNodes;
            selected = sorted.Take(maxNodes).ToList();
        }
        else if (_nodes.ContainsKey(nodeLabel))
        {
            // Degree-prioritized level BFS: at each depth, expand highest-degree nodes first.
            var bfsNodes = new List<string>();
            var visited = new HashSet<string>();
            var queue = new Queue<(string Node, int Depth, int Degree)>();
            queue.Enqueue((nodeLabel, 0, Degree(nodeLabel)));
            var hasUnexplored = false;

            while (queue.Count > 0 && bfsNodes.Count < maxNodes)
            {
                var currentDepth = queue.Peek().Depth;
                var level = new List<(string Node, int Depth, int Degree)>();
                while (queue.Count > 0 && queue.Peek().Depth == currentDepth)
                {
                    level.Add(queue.Dequeue());
                }
                level.Sort((a, b) => b.Degree.CompareTo(a.Degree)); // highest degree first

                foreach (var (node, depth, _) in level)
                {
                    if (visited.Add(node))
                    {
                        bfsNodes.Add(node);
                        var unvisited = (_adjacency.GetValueOrDefault(node) ?? []).Where(n => !visited.Contains(n)).ToList();
                        if (depth < maxDepth)
                        {
                            foreach (var nb in unvisited)
                            {
                                queue.Enqueue((nb, depth + 1, Degree(nb)));
                            }
                        }
                        else if (unvisited.Count > 0)
                        {
                            hasUnexplored = true;
                        }
                    }
                    if (bfsNodes.Count >= maxNodes)
                    {
                        break;
                    }
                }
            }

            // Truncation flag set only when the max_nodes cap is hit (matches Python).
            if (((queue.Count > 0 && bfsNodes.Count >= maxNodes) || hasUnexplored) && bfsNodes.Count >= maxNodes)
            {
                truncated = true;
            }
            selected = bfsNodes;
        }
        else
        {
            return Task.FromResult(new KnowledgeGraph());
        }

        var selectedSet = selected.ToHashSet();
        var graph = new KnowledgeGraph { IsTruncated = truncated };
        // Node/edge order follows the original graph's insertion order (networkx subgraph semantics).
        foreach (var id in _nodes.Keys)
        {
            if (!selectedSet.Contains(id))
            {
                continue;
            }
            graph.Nodes.Add(new KnowledgeGraphNode
            {
                Id = id,
                Labels = [id],
                Properties = new Dictionary<string, object?>(_nodes[id]),
            });
        }
        foreach (var (key, data) in _edges)
        {
            if (selectedSet.Contains(key.Item1) && selectedSet.Contains(key.Item2))
            {
                graph.Edges.Add(new KnowledgeGraphEdge
                {
                    Id = $"{key.Item1}-{key.Item2}",
                    Type = "DIRECTED",
                    Source = key.Item1,
                    Target = key.Item2,
                    Properties = new Dictionary<string, object?>(data),
                });
            }
        }
        return Task.FromResult(graph);
    }

    public Task IndexDoneCallbackAsync(CancellationToken cancellationToken = default)
    {
        if (_dirty)
        {
            Persist();
            _dirty = false;
        }
        return Task.CompletedTask;
    }

    public Task FinalizeAsync(CancellationToken cancellationToken = default) => IndexDoneCallbackAsync(cancellationToken);

    public Task<(string Status, string Message)> DropAsync(CancellationToken cancellationToken = default)
    {
        _nodes.Clear();
        _edges.Clear();
        _adjacency.Clear();
        _dirty = false;
        try
        {
            if (File.Exists(_filePath))
            {
                File.Delete(_filePath);
            }
            return Task.FromResult(("success", "data dropped"));
        }
        catch (Exception ex)
        {
            return Task.FromResult(("error", ex.Message));
        }
    }

    // ---- GraphML persistence ----

    private void Persist()
    {
        EnsureDirectory();

        var nodeAttrs = _nodes.Values.SelectMany(d => d.Keys).Distinct().ToList();
        var edgeAttrs = _edges.Values.SelectMany(d => d.Keys).Distinct().ToList();

        // Infer a GraphML type per attribute from its values (networkx write_graphml parity).
        var nodeTypes = nodeAttrs.ToDictionary(a => a, a => InferGraphmlType(_nodes.Values.Where(d => d.ContainsKey(a)).Select(d => d[a])));
        var edgeTypes = edgeAttrs.ToDictionary(a => a, a => InferGraphmlType(_edges.Values.Where(d => d.ContainsKey(a)).Select(d => d[a])));

        var graph = new XElement(Gml + "graph", new XAttribute("edgedefault", "undirected"));

        foreach (var (id, data) in _nodes)
        {
            var node = new XElement(Gml + "node", new XAttribute("id", id));
            foreach (var (attr, value) in data)
            {
                node.Add(new XElement(Gml + "data", new XAttribute("key", "n_" + attr), FormatGraphmlValue(value, nodeTypes[attr])));
            }
            graph.Add(node);
        }

        foreach (var (key, data) in _edges)
        {
            var edge = new XElement(Gml + "edge",
                new XAttribute("source", key.Item1),
                new XAttribute("target", key.Item2));
            foreach (var (attr, value) in data)
            {
                edge.Add(new XElement(Gml + "data", new XAttribute("key", "e_" + attr), FormatGraphmlValue(value, edgeTypes[attr])));
            }
            graph.Add(edge);
        }

        var root = new XElement(Gml + "graphml");
        foreach (var attr in nodeAttrs)
        {
            root.Add(new XElement(Gml + "key",
                new XAttribute("id", "n_" + attr), new XAttribute("for", "node"),
                new XAttribute("attr.name", attr), new XAttribute("attr.type", nodeTypes[attr])));
        }
        foreach (var attr in edgeAttrs)
        {
            root.Add(new XElement(Gml + "key",
                new XAttribute("id", "e_" + attr), new XAttribute("for", "edge"),
                new XAttribute("attr.name", attr), new XAttribute("attr.type", edgeTypes[attr])));
        }
        root.Add(graph);

        var doc = new XDocument(new XDeclaration("1.0", "utf-8", null), root);
        AtomicWrite(_filePath, doc.Declaration + Environment.NewLine + doc.ToString());
    }

    private void LoadFromDisk()
    {
        _nodes.Clear();
        _edges.Clear();
        _adjacency.Clear();
        if (!File.Exists(_filePath))
        {
            return;
        }

        var doc = XDocument.Load(_filePath);
        var root = doc.Root;
        if (root is null)
        {
            return;
        }

        // Map key id -> (attr.name, attr.type) so values coerce back to their declared type.
        var keyNames = new Dictionary<string, string>();
        var keyTypes = new Dictionary<string, string>();
        foreach (var k in root.Elements(Gml + "key"))
        {
            var keyId = k.Attribute("id")!.Value;
            keyNames[keyId] = k.Attribute("attr.name")?.Value ?? keyId;
            keyTypes[keyId] = k.Attribute("attr.type")?.Value ?? "string";
        }

        var graph = root.Element(Gml + "graph");
        if (graph is null)
        {
            return;
        }

        foreach (var node in graph.Elements(Gml + "node"))
        {
            var id = node.Attribute("id")!.Value;
            var data = new Dictionary<string, object?>();
            foreach (var d in node.Elements(Gml + "data"))
            {
                var keyId = d.Attribute("key")!.Value;
                data[keyNames.GetValueOrDefault(keyId, keyId)] = CoerceGraphmlValue(d.Value, keyTypes.GetValueOrDefault(keyId, "string"));
            }
            _nodes[id] = data;
            _adjacency.TryAdd(id, []);
        }

        foreach (var edge in graph.Elements(Gml + "edge"))
        {
            var source = edge.Attribute("source")!.Value;
            var target = edge.Attribute("target")!.Value;
            var data = new Dictionary<string, object?>();
            foreach (var d in edge.Elements(Gml + "data"))
            {
                var keyId = d.Attribute("key")!.Value;
                data[keyNames.GetValueOrDefault(keyId, keyId)] = CoerceGraphmlValue(d.Value, keyTypes.GetValueOrDefault(keyId, "string"));
            }
            _edges[Canon(source, target)] = data;
            _adjacency.TryAdd(source, []);
            _adjacency.TryAdd(target, []);
            _adjacency[source].Add(target);
            _adjacency[target].Add(source);
        }
    }

    // ---- GraphML typed-attribute helpers (networkx round-trip parity) ----

    /// <summary>Infer the GraphML attr.type ("boolean"/"long"/"double"/"string") from a column of values.</summary>
    private static string InferGraphmlType(IEnumerable<object?> values)
    {
        bool sawBool = false, sawLong = false, sawDouble = false, sawOther = false, any = false;
        foreach (var v in values)
        {
            if (v is null)
            {
                continue;
            }
            any = true;
            switch (v)
            {
                case bool: sawBool = true; break;
                case long or int or short or byte: sawLong = true; break;
                case double or float or decimal: sawDouble = true; break;
                default: sawOther = true; break;
            }
        }
        if (!any || sawOther)
        {
            return "string";
        }
        if (sawBool)
        {
            return sawLong || sawDouble ? "string" : "boolean";
        }
        if (sawDouble)
        {
            return "double";
        }
        return sawLong ? "long" : "string";
    }

    private static string FormatGraphmlValue(object? value, string type) => value switch
    {
        null => string.Empty,
        bool b => b ? "true" : "false",
        _ when type == "double" => Convert.ToDouble(value, CultureInfo.InvariantCulture).ToString("R", CultureInfo.InvariantCulture),
        _ when type == "long" => Convert.ToInt64(value, CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture),
        _ => value.ToString() ?? string.Empty,
    };

    private static object? CoerceGraphmlValue(string raw, string type) => type switch
    {
        "long" or "int" or "integer" when long.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var l) => l,
        "double" or "float" or "real" when double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var d) => d,
        "boolean" or "bool" => raw.Equals("true", StringComparison.OrdinalIgnoreCase)
            ? true
            : raw.Equals("false", StringComparison.OrdinalIgnoreCase) ? false : raw,
        _ => raw,
    };
}
