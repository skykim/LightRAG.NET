using LightRAG.Core.Abstractions;
using LightRAG.Core.Extraction;
using LightRAG.Core.Query;
using LightRAG.Core.Tokenization;
using LightRAG.Storage.FileBased;

namespace LightRAG.Core.Tests;

public class QueryEngineTests
{
    private static readonly TiktokenTokenizer Tokenizer = new("gpt-4o-mini");

    private static async Task<(GraphmlGraphStorage Graph, NanoVectorDbStorage Ent, NanoVectorDbStorage Rel, NanoVectorDbStorage Chunks, JsonKvStorage Text)>
        SeedAsync(string dir)
    {
        var emb = FakeEmbedding.Create();
        var graph = new GraphmlGraphStorage(dir, emb);
        var ent = new NanoVectorDbStorage(dir, NameSpace.VectorStoreEntities, emb, 0f);
        var rel = new NanoVectorDbStorage(dir, NameSpace.VectorStoreRelationships, emb, 0f);
        var chunks = new NanoVectorDbStorage(dir, NameSpace.VectorStoreChunks, emb, 0f);
        var text = new JsonKvStorage(dir, NameSpace.KvStoreTextChunks);
        foreach (var s in new IStorageNameSpace[] { graph, ent, rel, chunks, text })
        {
            await s.InitializeAsync();
        }

        // Source chunk text.
        await text.UpsertAsync(new Dictionary<string, StorageRecord>
        {
            ["chunk-1"] = new() { ["content"] = "Alice works at Acme as an engineer.", ["file_path"] = "doc.txt" },
        });

        // Chunk vector (for naive/mix).
        await chunks.UpsertAsync(new Dictionary<string, StorageRecord>
        {
            ["chunk-1"] = new() { ["content"] = "Alice works at Acme as an engineer.", ["file_path"] = "doc.txt" },
        });

        // Knowledge graph via the builder.
        var r = new ChunkExtractionResult();
        r.Nodes["Alice"] = [new ExtractedNode { EntityName = "Alice", EntityType = "person", Description = "Alice is an engineer at Acme.", SourceId = "chunk-1", FilePath = "doc.txt", Timestamp = 1 }];
        r.Nodes["Acme"] = [new ExtractedNode { EntityName = "Acme", EntityType = "organization", Description = "Acme is a company that employs Alice.", SourceId = "chunk-1", FilePath = "doc.txt", Timestamp = 1 }];
        r.Edges[("Alice", "Acme")] = [new ExtractedEdge { SrcId = "Alice", TgtId = "Acme", Weight = 1, Description = "Alice works at Acme.", Keywords = "employment", SourceId = "chunk-1", FilePath = "doc.txt", Timestamp = 1 }];
        var builder = new KnowledgeGraphBuilder(graph, ent, rel, Tokenizer, new FakeLlmCaller());
        await builder.MergeAsync([r]);
        foreach (var s in new IStorageNameSpace[] { graph, ent, rel, chunks, text })
        {
            await s.IndexDoneCallbackAsync();
        }
        return (graph, ent, rel, chunks, text);
    }

    [Fact]
    public async Task Local_mode_retrieves_entities_and_answers()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var (graph, ent, rel, chunks, text) = await SeedAsync(dir);
            var keywordLlm = new FakeLlmCaller("""{"high_level_keywords": ["employment"], "low_level_keywords": ["Alice", "Acme"]}""");
            var queryLlm = new FakeLlmCaller("Alice works at Acme as an engineer.");

            var engine = new QueryEngine(graph, ent, rel, chunks, text, Tokenizer, queryLlm, keywordLlm);
            var result = await engine.QueryAsync("Where does Alice work?", new QueryParam { Mode = QueryMode.Local });

            Assert.Equal("Alice works at Acme as an engineer.", result.Content);
            // Keyword extraction happened, then the answer call.
            Assert.Single(keywordLlm.Calls);
            Assert.Single(queryLlm.Calls);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task OnlyNeedContext_returns_context_with_entities_and_chunks()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var (graph, ent, rel, chunks, text) = await SeedAsync(dir);
            var keywordLlm = new FakeLlmCaller("""{"high_level_keywords": ["employment"], "low_level_keywords": ["Alice"]}""");
            var queryLlm = new FakeLlmCaller("(unused)");

            var engine = new QueryEngine(graph, ent, rel, chunks, text, Tokenizer, queryLlm, keywordLlm);
            var result = await engine.QueryAsync("Where does Alice work?",
                new QueryParam { Mode = QueryMode.Hybrid, OnlyNeedContext = true });

            Assert.Contains("Alice", result.Content);
            Assert.Contains("Acme", result.Content);
            Assert.Contains("works at Acme", result.Content); // source chunk text included
            Assert.Empty(queryLlm.Calls); // no answer generation when only_need_context
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task Naive_mode_uses_chunk_vectors_only()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var (graph, ent, rel, chunks, text) = await SeedAsync(dir);
            var keywordLlm = new FakeLlmCaller("(unused)");
            var queryLlm = new FakeLlmCaller("Alice is an engineer.");

            var engine = new QueryEngine(graph, ent, rel, chunks, text, Tokenizer, queryLlm, keywordLlm);
            var result = await engine.QueryAsync("engineer", new QueryParam { Mode = QueryMode.Naive });

            Assert.Equal("Alice is an engineer.", result.Content);
            Assert.Empty(keywordLlm.Calls); // naive mode does not extract keywords
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task Bypass_mode_calls_llm_directly()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var (graph, ent, rel, chunks, text) = await SeedAsync(dir);
            var queryLlm = new FakeLlmCaller("direct answer");
            var engine = new QueryEngine(graph, ent, rel, chunks, text, Tokenizer, queryLlm, new FakeLlmCaller());

            var result = await engine.QueryAsync("hello", new QueryParam { Mode = QueryMode.Bypass });
            Assert.Equal("direct answer", result.Content);
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---- C-1: graph 1-hop expansion ----

    [Fact]
    public async Task C1_local_mode_includes_related_relations()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var (graph, ent, rel, chunks, text) = await SeedAsync(dir);
            var keywordLlm = new FakeLlmCaller("""{"high_level_keywords": ["employment"], "low_level_keywords": ["Alice", "Acme"]}""");
            var engine = new QueryEngine(graph, ent, rel, chunks, text, Tokenizer, new FakeLlmCaller("x"), keywordLlm);

            var result = await engine.QueryAsync("Where does Alice work?",
                new QueryParam { Mode = QueryMode.Local, OnlyNeedContext = true });

            // Before C-1, pure local returned NO relations; now the entities' incident edge is expanded in.
            Assert.Contains("entity1", result.Content);
            Assert.Contains("Alice works at Acme", result.Content);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task C1_global_mode_includes_endpoint_entities()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var (graph, ent, rel, chunks, text) = await SeedAsync(dir);
            var keywordLlm = new FakeLlmCaller("""{"high_level_keywords": ["employment"], "low_level_keywords": ["Alice"]}""");
            var engine = new QueryEngine(graph, ent, rel, chunks, text, Tokenizer, new FakeLlmCaller("x"), keywordLlm);

            var result = await engine.QueryAsync("Who is employed?",
                new QueryParam { Mode = QueryMode.Global, OnlyNeedContext = true });

            // Before C-1, pure global returned NO entities; now the relation's endpoint entities are expanded in.
            Assert.Contains("\"type\":", result.Content); // entity objects carry a "type" field; relations do not
            Assert.Contains("Acme", result.Content);
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---- M-7: Python-style compact context JSON (created_at present, no synthetic id) ----

    [Fact]
    public async Task M7_context_uses_python_style_compact_json()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var (graph, ent, rel, chunks, text) = await SeedAsync(dir);
            var keywordLlm = new FakeLlmCaller("""{"high_level_keywords": ["employment"], "low_level_keywords": ["Alice"]}""");
            var engine = new QueryEngine(graph, ent, rel, chunks, text, Tokenizer, new FakeLlmCaller("x"), keywordLlm);

            var result = await engine.QueryAsync("Where does Alice work?",
                new QueryParam { Mode = QueryMode.Hybrid, OnlyNeedContext = true });

            Assert.Contains("{\"entity\": \"", result.Content);   // compact object with Python ", "/": " spacing
            Assert.Contains("\"created_at\":", result.Content);   // M-7 restores created_at field
            Assert.DoesNotContain("\"id\":", result.Content);     // M-7 drops the synthetic id field
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---- C-3: EnableRerank is now live (reranker + min_rerank_score filtering) ----

    [Fact]
    public async Task C3_reranker_filters_chunks_below_min_score()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var emb = FakeEmbedding.Create();
            var graph = new GraphmlGraphStorage(dir, emb);
            var ent = new NanoVectorDbStorage(dir, NameSpace.VectorStoreEntities, emb, 0f);
            var rel = new NanoVectorDbStorage(dir, NameSpace.VectorStoreRelationships, emb, 0f);
            var chunks = new NanoVectorDbStorage(dir, NameSpace.VectorStoreChunks, emb, 0f);
            var text = new JsonKvStorage(dir, NameSpace.KvStoreTextChunks);
            foreach (var s in new IStorageNameSpace[] { graph, ent, rel, chunks, text }) { await s.InitializeAsync(); }

            await chunks.UpsertAsync(new Dictionary<string, StorageRecord>
            {
                ["c1"] = new() { ["content"] = "keep this one", ["file_path"] = "a.txt" },
                ["c2"] = new() { ["content"] = "drop this chunk", ["file_path"] = "b.txt" },
                ["c3"] = new() { ["content"] = "keep that too", ["file_path"] = "c.txt" },
            });
            foreach (var s in new IStorageNameSpace[] { chunks }) { await s.IndexDoneCallbackAsync(); }

            // Reranker scores "keep"-containing chunks high; min score filters the rest out.
            var options = new QueryOptions
            {
                MinRerankScore = 0.5,
                Reranker = (q, contents, _) =>
                    Task.FromResult<IReadOnlyList<double>>(contents.Select(c => c.Contains("keep") ? 0.9 : 0.1).ToList()),
            };
            var engine = new QueryEngine(graph, ent, rel, chunks, text, Tokenizer, new FakeLlmCaller("x"), new FakeLlmCaller(), options);

            var result = await engine.QueryAsync("anything",
                new QueryParam { Mode = QueryMode.Naive, OnlyNeedContext = true, EnableRerank = true });

            Assert.Contains("keep this one", result.Content);
            Assert.Contains("keep that too", result.Content);
            Assert.DoesNotContain("drop this chunk", result.Content); // filtered by min_rerank_score
        }
        finally { Directory.Delete(dir, true); }
    }
}
