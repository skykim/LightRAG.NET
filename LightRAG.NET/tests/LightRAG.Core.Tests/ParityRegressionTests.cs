using System.Runtime.CompilerServices;
using System.Text;
using LightRAG.Core.Abstractions;
using LightRAG.Core.Concurrency;
using LightRAG.Core.Extraction;
using LightRAG.Core.Llm;
using LightRAG.Core.Prompts;
using LightRAG.Core.Tokenization;
using LightRAG.Core.Utils;
using LightRAG.Storage.FileBased;

namespace LightRAG.Core.Tests;

/// <summary>A counting ILlmModel with a configurable model name, for cache-key tests.</summary>
internal sealed class CountingLlmModel(string name, string response) : ILlmModel
{
    public int Calls { get; private set; }
    public string ModelName => name;

    public Task<string> CompleteAsync(string prompt, LlmCompletionOptions? options = null, CancellationToken cancellationToken = default)
    {
        Calls++;
        return Task.FromResult(response);
    }

    public async IAsyncEnumerable<string> StreamAsync(string prompt, LlmCompletionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        yield return await CompleteAsync(prompt, options, cancellationToken).ConfigureAwait(false);
    }
}

/// <summary>
/// Regression tests for the Python-parity fixes catalogued in PARITY_REVIEW.md
/// (C-1..C-5 critical, M-1..M-13 major). Each test pins a behavior that previously
/// diverged from <c>lightrag</c> (Python).
/// </summary>
public class ParityRegressionTests
{
    private static readonly TiktokenTokenizer Tokenizer = new("gpt-4o-mini");
    private const string D = PromptTemplates.DefaultTupleDelimiter;
    private const string C = PromptTemplates.DefaultCompletionDelimiter;

    // ---- M-1: record-recovery pre-pass (glued records) ----

    [Fact]
    public async Task M1_recovers_entity_and_relation_glued_on_one_line()
    {
        // The LLM emitted an entity and a relation on a single line, separated by the tuple
        // delimiter instead of a newline. Without the fixed_records recovery pass the whole
        // line is one 9-field record and both are dropped.
        var output = $"entity{D}Alice{D}Person{D}Alice is an engineer.{D}relation{D}Alice{D}Acme{D}employment{D}Alice works at Acme.\n{C}";
        var extractor = new EntityExtractor(new FakeLlmCaller(output), Tokenizer,
            new ExtractionOptions { MaxGleaning = 0 });

        var results = await extractor.ExtractAsync([new ChunkInput("chunk-1", "t", "doc.txt")]);
        var r = results[0];

        Assert.True(r.Nodes.ContainsKey("Alice"), "entity recovered from glued line");
        Assert.True(r.Edges.ContainsKey(("Alice", "Acme")), "relation recovered from glued line");
    }

    // ---- M-2: tuple-delimiter corruption repair ----

    [Theory]
    [InlineData("<|##|>")]   // doubled core
    [InlineData("<|#>")]     // missing closing pipe
    [InlineData("<#|>")]     // missing opening pipe
    [InlineData("<|>")]      // core dropped entirely
    public async Task M2_repairs_corrupted_tuple_delimiter(string corrupt)
    {
        // First delimiter of the entity record is corrupted; the repair battery must normalize it
        // so the record splits into the expected 4 fields.
        var output = $"entity{corrupt}Alice{D}Person{D}Alice is an engineer.\n{C}";
        var extractor = new EntityExtractor(new FakeLlmCaller(output), Tokenizer,
            new ExtractionOptions { MaxGleaning = 0 });

        var results = await extractor.ExtractAsync([new ChunkInput("chunk-1", "t", "doc.txt")]);
        Assert.True(results[0].Nodes.ContainsKey("Alice"), $"entity recovered after repairing '{corrupt}'");
    }

    // ---- M-3: entity-name UTF-8 byte limit ----

    [Fact]
    public async Task M3_truncates_cjk_entity_name_to_utf8_byte_limit()
    {
        // 200 CJK chars = 600 UTF-8 bytes: within the 256-char limit but over the 512-byte limit.
        var longName = new string('中', 200); // 中 * 200
        var output = $"entity{D}{longName}{D}Person{D}A long-named entity.\n{C}";
        var extractor = new EntityExtractor(new FakeLlmCaller(output), Tokenizer,
            new ExtractionOptions { MaxGleaning = 0 });

        var results = await extractor.ExtractAsync([new ChunkInput("chunk-1", "t", "doc.txt")]);
        var key = Assert.Single(results[0].Nodes).Key;

        Assert.True(Encoding.UTF8.GetByteCount(key) <= 512, "name fits the 512-byte limit");
        Assert.Equal(170, key.Length);          // 170 * 3 bytes = 510 <= 512
        Assert.All(key, c => Assert.Equal('中', c)); // still valid UTF-8, no broken char
    }

    // ---- C-4: map-reduce summarization (no silent description loss) ----

    [Fact]
    public async Task C4_map_reduce_summarization_makes_multiple_llm_calls()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var emb = FakeEmbedding.Create();
            var graph = new GraphmlGraphStorage(dir, emb);
            var entityVdb = new NanoVectorDbStorage(dir, NameSpace.VectorStoreEntities, emb, cosineBetterThanThreshold: 0f);
            var relVdb = new NanoVectorDbStorage(dir, NameSpace.VectorStoreRelationships, emb, cosineBetterThanThreshold: 0f);
            await graph.InitializeAsync();
            await entityVdb.InitializeAsync();
            await relVdb.InitializeAsync();

            // Six distinct descriptions for one entity; a tiny context size forces the map phase
            // to split them into several chunks instead of a single truncating LLM call.
            var node = new ChunkExtractionResult();
            node.Nodes["Alice"] = Enumerable.Range(1, 6).Select(i => new ExtractedNode
            {
                EntityName = "Alice",
                EntityType = "person",
                Description = $"Alice is a software engineer who works on distributed systems and databases, fact number {i}.",
                SourceId = $"chunk-{i}",
                FilePath = "doc.txt",
                Timestamp = i,
            }).ToList();

            var llm = new FakeLlmCaller(Enumerable.Repeat("Summarized.", 30).ToArray());
            var options = new MergeOptions { SummaryContextSize = 50, SummaryMaxTokens = 10, ForceLlmSummaryOnMerge = 2 };
            var builder = new KnowledgeGraphBuilder(graph, entityVdb, relVdb, Tokenizer, llm, options);

            await builder.MergeAsync([node]);

            // Old behavior: exactly ONE truncating LLM call (overflow silently dropped).
            // Map-reduce: multiple calls across map + reduce phases.
            Assert.True(llm.Calls.Count > 1, $"expected multi-call map-reduce, got {llm.Calls.Count} call(s)");

            var stored = await graph.GetNodeAsync("Alice");
            Assert.False(string.IsNullOrEmpty(stored!.GetString("description")));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task C4_small_description_set_joins_without_llm()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var emb = FakeEmbedding.Create();
            var graph = new GraphmlGraphStorage(dir, emb);
            var entityVdb = new NanoVectorDbStorage(dir, NameSpace.VectorStoreEntities, emb, cosineBetterThanThreshold: 0f);
            var relVdb = new NanoVectorDbStorage(dir, NameSpace.VectorStoreRelationships, emb, cosineBetterThanThreshold: 0f);
            await graph.InitializeAsync();
            await entityVdb.InitializeAsync();
            await relVdb.InitializeAsync();

            var node = new ChunkExtractionResult();
            node.Nodes["Alice"] =
            [
                new ExtractedNode { EntityName = "Alice", EntityType = "person", Description = "Alice is an engineer.", SourceId = "c1", FilePath = "d", Timestamp = 1 },
                new ExtractedNode { EntityName = "Alice", EntityType = "person", Description = "Alice leads a team.", SourceId = "c2", FilePath = "d", Timestamp = 2 },
            ];

            var llm = new FakeLlmCaller(); // no responses queued: any call would return ""
            var builder = new KnowledgeGraphBuilder(graph, entityVdb, relVdb, Tokenizer, llm);
            await builder.MergeAsync([node]);

            Assert.Empty(llm.Calls); // < force_llm_summary_on_merge and under token budget -> join only
            var desc = (await graph.GetNodeAsync("Alice"))!.GetString("description")!;
            Assert.Contains("engineer", desc);
            Assert.Contains("leads a team", desc);
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---- M-9: empty-description relation raises (vs fabricating) ----

    [Fact]
    public async Task M9_relation_without_description_throws()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var emb = FakeEmbedding.Create();
            var graph = new GraphmlGraphStorage(dir, emb);
            var entityVdb = new NanoVectorDbStorage(dir, NameSpace.VectorStoreEntities, emb, cosineBetterThanThreshold: 0f);
            var relVdb = new NanoVectorDbStorage(dir, NameSpace.VectorStoreRelationships, emb, cosineBetterThanThreshold: 0f);
            await graph.InitializeAsync();
            await entityVdb.InitializeAsync();
            await relVdb.InitializeAsync();

            var result = new ChunkExtractionResult();
            result.Edges[("A", "B")] =
            [
                new ExtractedEdge { SrcId = "A", TgtId = "B", Weight = 1.0, Description = "", Keywords = "k", SourceId = "c1", FilePath = "d", Timestamp = 1 },
            ];

            var builder = new KnowledgeGraphBuilder(graph, entityVdb, relVdb, Tokenizer, new FakeLlmCaller());
            await Assert.ThrowsAsync<InvalidOperationException>(() => builder.MergeAsync([result]));
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---- C-5: doc-status persisted immediately on upsert ----

    [Fact]
    public async Task C5_doc_status_upsert_persists_immediately_without_explicit_flush()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var store = new JsonDocStatusStorage(dir);
            await store.InitializeAsync();
            await store.UpsertAsync(new Dictionary<string, StorageRecord>
            {
                ["d1"] = new() { ["status"] = "processing", ["content_summary"] = "x", ["content_length"] = 10, ["file_path"] = "a.txt", ["created_at"] = "t", ["updated_at"] = "t" },
            });
            // Intentionally NO IndexDoneCallbackAsync: upsert itself must have flushed (crash-recovery anchor).

            var reopened = new JsonDocStatusStorage(dir);
            await reopened.InitializeAsync();
            var processing = await reopened.GetDocsByStatusAsync(DocStatus.Processing);
            Assert.True(processing.ContainsKey("d1"), "doc-status survived a simulated crash before batch flush");

            // chunks_list is defaulted by upsert (matches Python pre-processing).
            var record = await reopened.GetByIdAsync("d1");
            Assert.True(record!.ContainsKey("chunks_list"));
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---- M-5: flattened, model-partitioned LLM cache key ----

    [Fact]
    public async Task M5_cache_hits_on_repeat_and_uses_flattened_key()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var cache = new JsonKvStorage(dir, NameSpace.KvStoreLlmResponseCache);
            await cache.InitializeAsync();
            var scheduler = new PriorityAsyncScheduler(4);
            var model = new CountingLlmModel("model-a", "answer-A");
            var caller = new CachedLlmCaller(model, scheduler, cache, role: "extract");

            var first = await caller.CompleteAsync("prompt", cacheType: "extract");
            var second = await caller.CompleteAsync("prompt", cacheType: "extract");

            Assert.Equal("answer-A", first);
            Assert.Equal("answer-A", second);
            Assert.Equal(1, model.Calls); // second call served from cache

            // The stored key is the flattened "default:{cache_type}:{md5}" form.
            await cache.IndexDoneCallbackAsync();
            var json = await File.ReadAllTextAsync(Path.Combine(dir, "kv_store_llm_response_cache.json"));
            Assert.Contains("default:extract:", json);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task M5_model_swap_invalidates_cache()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var cache = new JsonKvStorage(dir, NameSpace.KvStoreLlmResponseCache);
            await cache.InitializeAsync();
            var scheduler = new PriorityAsyncScheduler(4);

            var modelA = new CountingLlmModel("model-a", "answer-A");
            var callerA = new CachedLlmCaller(modelA, scheduler, cache, role: "extract");
            var a = await callerA.CompleteAsync("prompt", cacheType: "extract");

            // A different model shares the cache but must NOT see model-a's entry (identity-partitioned key).
            var modelB = new CountingLlmModel("model-b", "answer-B");
            var callerB = new CachedLlmCaller(modelB, scheduler, cache, role: "extract");
            var b = await callerB.CompleteAsync("prompt", cacheType: "extract");

            Assert.Equal("answer-A", a);
            Assert.Equal("answer-B", b);          // fresh call, not the stale "answer-A"
            Assert.Equal(1, modelB.Calls);        // model-b had a genuine cache miss
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---- M-13: DocStatus counts zero-seed every known status ----

    [Fact]
    public async Task M13_status_counts_zero_seed_all_statuses()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var store = new JsonDocStatusStorage(dir);
            await store.InitializeAsync();
            await store.UpsertAsync(new Dictionary<string, StorageRecord>
            {
                ["d1"] = new() { ["status"] = "processed", ["content_summary"] = "x", ["content_length"] = 1, ["file_path"] = "a", ["created_at"] = "t", ["updated_at"] = "t" },
            });

            var counts = await store.GetStatusCountsAsync();
            foreach (var s in new[] { "pending", "parsing", "analyzing", "processing", "preprocessed", "processed", "failed" })
            {
                Assert.True(counts.ContainsKey(s), $"status '{s}' present in counts");
            }
            Assert.Equal(1, counts["processed"]);
            Assert.Equal(0, counts["pending"]);
            Assert.Equal(0, counts["parsing"]);
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---- M-12: VDB persists only whitelisted meta fields ----

    [Fact]
    public async Task M12_vdb_persists_only_whitelisted_meta_fields()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var vdb = new NanoVectorDbStorage(dir, NameSpace.VectorStoreEntities, FakeEmbedding.Create(), 0f,
                metaFields: ["entity_name", "source_id", "content", "file_path"]);
            await vdb.InitializeAsync();
            var id = Hashing.ComputeMdHashId("Alice", "ent-");
            await vdb.UpsertAsync(new Dictionary<string, StorageRecord>
            {
                [id] = new()
                {
                    ["entity_name"] = "Alice",
                    ["entity_type"] = "person",   // NOT whitelisted -> must be dropped
                    ["content"] = "Alice\nAlice is an engineer.",
                    ["source_id"] = "chunk-1",
                    ["file_path"] = "doc.txt",
                },
            });
            await vdb.IndexDoneCallbackAsync();

            var record = await vdb.GetByIdAsync(id);
            Assert.NotNull(record);
            Assert.True(record!.ContainsKey("entity_name"));
            Assert.True(record.ContainsKey("content"));
            Assert.False(record.ContainsKey("entity_type")); // filtered by meta_fields whitelist
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---- M-6: identical content with different filenames is deduplicated ----

    [Fact]
    public async Task M6_same_content_different_filename_is_deduplicated()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var rag = FileBasedLightRag.Create(dir, new FakeLlmModel(), FakeEmbedding.Create(), cosineThreshold: 0f);
            await rag.InitializeAsync();

            const string content = "Alice is a software engineer at Acme.";
            Assert.Equal(1, await rag.InsertAsync(content, "first.txt"));
            Assert.Equal(0, await rag.InsertAsync(content, "second.txt")); // same content, different name -> dedup
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---- M-4: JSON-mode extraction ships three few-shot examples by default ----

    [Fact]
    public void M4_json_examples_match_python_three_example_block()
    {
        Assert.Equal(3, PromptTemplates.EntityExtractionJsonExamples.Length);
        Assert.Equal(3, new ExtractionOptions().JsonExamples.Count); // defaulted, not empty

        var joined = string.Join("\n", PromptTemplates.EntityExtractionJsonExamples);
        Assert.Contains("Bornean Orangutan", joined);
        Assert.Contains("NASBench-360", joined);
        Assert.Contains("\"name\": \"Alex\"", joined);
    }

    // ---- M-11: get_knowledge_graph degree-priority BFS + DIRECTED edges ----

    [Fact]
    public async Task M11_knowledge_graph_prioritizes_high_degree_and_marks_directed()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var graph = new GraphmlGraphStorage(dir, FakeEmbedding.Create());
            await graph.InitializeAsync();
            await graph.UpsertNodeAsync("Hub", new Dictionary<string, object?> { ["entity_type"] = "x" });
            foreach (var n in new[] { "A", "B", "C" })
            {
                await graph.UpsertNodeAsync(n, new Dictionary<string, object?> { ["entity_type"] = "x" });
                await graph.UpsertEdgeAsync("Hub", n, new Dictionary<string, object?> { ["description"] = "rel" });
            }

            // "*" capped at 2 -> the highest-degree node (Hub) is kept and the result is truncated.
            var capped = await graph.GetKnowledgeGraphAsync("*", maxNodes: 2);
            Assert.True(capped.IsTruncated);
            Assert.Equal(2, capped.Nodes.Count);
            Assert.Contains(capped.Nodes, node => node.Id == "Hub");

            // Edges are typed DIRECTED, matching networkx get_knowledge_graph.
            var full = await graph.GetKnowledgeGraphAsync("*", maxNodes: 100);
            Assert.NotEmpty(full.Edges);
            Assert.All(full.Edges, e => Assert.Equal("DIRECTED", e.Type));
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---- M-10: GraphML preserves attribute types (long / double / boolean) across a disk round-trip ----

    [Fact]
    public async Task M10_graphml_preserves_attribute_types_on_roundtrip()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var g = new GraphmlGraphStorage(dir, FakeEmbedding.Create());
            await g.InitializeAsync();
            await g.UpsertNodeAsync("Alice", new Dictionary<string, object?> { ["entity_type"] = "Person", ["created_at"] = 1717000000L });
            await g.UpsertNodeAsync("Acme", new Dictionary<string, object?> { ["entity_type"] = "Org", ["created_at"] = 1717000001L });
            await g.UpsertEdgeAsync("Alice", "Acme", new Dictionary<string, object?> { ["weight"] = 2.5, ["verified"] = true, ["created_at"] = 1717000002L });
            await g.IndexDoneCallbackAsync();

            var g2 = new GraphmlGraphStorage(dir, FakeEmbedding.Create());
            await g2.InitializeAsync();

            var node = await g2.GetNodeAsync("Alice");
            Assert.IsType<long>(node!["created_at"]);          // numeric, not a bare string

            var edge = await g2.GetEdgeAsync("Acme", "Alice"); // reversed lookup (undirected)
            Assert.IsType<double>(edge!["weight"]);
            Assert.Equal(2.5, Convert.ToDouble(edge["weight"]));
            Assert.IsType<bool>(edge["verified"]);
            Assert.True((bool)edge["verified"]!);
            Assert.Equal("Person", node["entity_type"]);       // strings still round-trip as strings
        }
        finally { Directory.Delete(dir, true); }
    }
}
