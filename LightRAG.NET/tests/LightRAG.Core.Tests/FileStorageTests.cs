using LightRAG.Core.Abstractions;
using LightRAG.Storage.FileBased;

namespace LightRAG.Core.Tests;

/// <summary>A deterministic, offline embedding function for tests: hashes tokens into a fixed-dim bag-of-words vector.</summary>
public static class FakeEmbedding
{
    public const int Dim = 16;

    public static EmbeddingFunc Create() => new(
        EmbeddingDim: Dim,
        Func: (texts, _, _) =>
        {
            var vectors = texts.Select(Embed).ToArray();
            return Task.FromResult(vectors);
        });

    private static float[] Embed(string text)
    {
        var vec = new float[Dim];
        foreach (var token in text.ToLowerInvariant().Split([' ', ',', '.', '!', '?', '\n'], StringSplitOptions.RemoveEmptyEntries))
        {
            vec[StableHash(token) % Dim] += 1f;
        }
        return vec;
    }

    // FNV-1a: deterministic across processes (unlike string.GetHashCode, which is randomized).
    private static int StableHash(string s)
    {
        uint hash = 2166136261;
        foreach (var c in s)
        {
            hash = (hash ^ c) * 16777619;
        }
        return (int)(hash & 0x7fffffff);
    }
}

public class JsonKvStorageTests
{
    [Fact]
    public async Task Upsert_persist_reload_roundtrips()
    {
        var dir = NewTempDir();
        try
        {
            var kv = new JsonKvStorage(dir, NameSpace.KvStoreTextChunks);
            await kv.InitializeAsync();
            await kv.UpsertAsync(new Dictionary<string, StorageRecord>
            {
                ["c1"] = new() { ["content"] = "hello", ["tokens"] = 3 },
            });
            await kv.IndexDoneCallbackAsync();

            // Reload into a fresh instance from disk.
            var kv2 = new JsonKvStorage(dir, NameSpace.KvStoreTextChunks);
            await kv2.InitializeAsync();
            var record = await kv2.GetByIdAsync("c1");
            Assert.NotNull(record);
            Assert.Equal("hello", record!["content"]);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task FilterKeys_returns_missing()
    {
        var dir = NewTempDir();
        try
        {
            var kv = new JsonKvStorage(dir, NameSpace.KvStoreFullDocs);
            await kv.InitializeAsync();
            await kv.UpsertAsync(new Dictionary<string, StorageRecord> { ["a"] = new() { ["x"] = 1 } });
            var missing = await kv.FilterKeysAsync(new HashSet<string> { "a", "b", "c" });
            Assert.Equal(["b", "c"], missing.OrderBy(x => x));
        }
        finally { Directory.Delete(dir, true); }
    }

    internal static string NewTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "lightrag-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }
}

public class NanoVectorDbStorageTests
{
    [Fact]
    public async Task Ranks_by_cosine_similarity()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var vdb = new NanoVectorDbStorage(dir, NameSpace.VectorStoreChunks, FakeEmbedding.Create(), cosineBetterThanThreshold: 0f);
            await vdb.InitializeAsync();
            await vdb.UpsertAsync(new Dictionary<string, StorageRecord>
            {
                ["c1"] = new() { ["content"] = "the cat sat on the mat" },
                ["c2"] = new() { ["content"] = "quantum chromodynamics lagrangian" },
                ["c3"] = new() { ["content"] = "a cat and a dog" },
            });
            await vdb.IndexDoneCallbackAsync();

            var results = await vdb.QueryAsync("cat", topK: 2);
            Assert.Equal(2, results.Count);
            // The two cat-related chunks should rank above the physics one.
            var topIds = results.Select(r => r["id"]).ToHashSet();
            Assert.Contains("c1", topIds);
            Assert.Contains("c3", topIds);
            Assert.DoesNotContain("c2", topIds);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task Persists_and_reloads_vectors()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var vdb = new NanoVectorDbStorage(dir, NameSpace.VectorStoreEntities, FakeEmbedding.Create(), cosineBetterThanThreshold: 0f);
            await vdb.InitializeAsync();
            await vdb.UpsertAsync(new Dictionary<string, StorageRecord> { ["e1"] = new() { ["content"] = "alpha beta" } });
            await vdb.IndexDoneCallbackAsync();

            var vdb2 = new NanoVectorDbStorage(dir, NameSpace.VectorStoreEntities, FakeEmbedding.Create(), cosineBetterThanThreshold: 0f);
            await vdb2.InitializeAsync();
            var results = await vdb2.QueryAsync("alpha", topK: 1);
            Assert.Single(results);
            Assert.Equal("e1", results[0]["id"]);
        }
        finally { Directory.Delete(dir, true); }
    }
}

public class GraphmlGraphStorageTests
{
    [Fact]
    public async Task Upsert_and_query_undirected_edges()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var graph = new GraphmlGraphStorage(dir, FakeEmbedding.Create());
            await graph.InitializeAsync();
            await graph.UpsertNodeAsync("Alice", new Dictionary<string, object?> { ["entity_type"] = "Person", ["description"] = "An engineer" });
            await graph.UpsertNodeAsync("Acme", new Dictionary<string, object?> { ["entity_type"] = "Organization" });
            await graph.UpsertEdgeAsync("Alice", "Acme", new Dictionary<string, object?> { ["description"] = "works at" });

            // Undirected: edge is found in both directions.
            Assert.True(await graph.HasEdgeAsync("Alice", "Acme"));
            Assert.True(await graph.HasEdgeAsync("Acme", "Alice"));
            Assert.Equal(1, await graph.NodeDegreeAsync("Alice"));
            Assert.Equal(2, await graph.EdgeDegreeAsync("Alice", "Acme"));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task GraphML_roundtrips_through_disk()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var graph = new GraphmlGraphStorage(dir, FakeEmbedding.Create());
            await graph.InitializeAsync();
            await graph.UpsertNodeAsync("Alice", new Dictionary<string, object?> { ["entity_type"] = "Person" });
            await graph.UpsertNodeAsync("Bob", new Dictionary<string, object?> { ["entity_type"] = "Person" });
            await graph.UpsertEdgeAsync("Alice", "Bob", new Dictionary<string, object?> { ["description"] = "knows", ["weight"] = "1.0" });
            await graph.IndexDoneCallbackAsync();

            var graph2 = new GraphmlGraphStorage(dir, FakeEmbedding.Create());
            await graph2.InitializeAsync();
            var node = await graph2.GetNodeAsync("Alice");
            Assert.Equal("Person", node!["entity_type"]);
            var edge = await graph2.GetEdgeAsync("Bob", "Alice"); // reversed lookup
            Assert.NotNull(edge);
            Assert.Equal("knows", edge!["description"]);
        }
        finally { Directory.Delete(dir, true); }
    }
}

public class JsonDocStatusStorageTests
{
    [Fact]
    public async Task Filters_docs_by_status()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var store = new JsonDocStatusStorage(dir);
            await store.InitializeAsync();
            await store.UpsertAsync(new Dictionary<string, StorageRecord>
            {
                ["d1"] = new() { ["status"] = "pending", ["content_summary"] = "doc one", ["content_length"] = 100, ["file_path"] = "a.txt", ["created_at"] = "t", ["updated_at"] = "t" },
                ["d2"] = new() { ["status"] = "processed", ["content_summary"] = "doc two", ["content_length"] = 200, ["file_path"] = "b.txt", ["created_at"] = "t", ["updated_at"] = "t" },
            });
            await store.IndexDoneCallbackAsync();

            var pending = await store.GetDocsByStatusAsync(DocStatus.Pending);
            Assert.Single(pending);
            Assert.True(pending.ContainsKey("d1"));
            Assert.Equal(100, pending["d1"].ContentLength);

            var counts = await store.GetStatusCountsAsync();
            Assert.Equal(1, counts["pending"]);
            Assert.Equal(1, counts["processed"]);
        }
        finally { Directory.Delete(dir, true); }
    }
}
