using System.Runtime.CompilerServices;
using LightRAG.Core;
using LightRAG.Core.Abstractions;
using LightRAG.Core.Prompts;
using LightRAG.Storage.FileBased;

namespace LightRAG.Core.Tests;

/// <summary>An offline ILlmModel that returns role-appropriate canned output based on the prompt shape.</summary>
internal sealed class FakeLlmModel : ILlmModel
{
    private const string D = PromptTemplates.DefaultTupleDelimiter;
    private const string C = PromptTemplates.DefaultCompletionDelimiter;

    public string ModelName => "fake";

    public Task<string> CompleteAsync(string prompt, LlmCompletionOptions? options = null, CancellationToken cancellationToken = default)
    {
        var system = options?.SystemPrompt ?? string.Empty;

        // Keyword extraction (query path): JSON mode + keyword extractor prompt.
        if (options?.JsonMode == true && prompt.Contains("keyword extractor"))
        {
            return Task.FromResult("""{"high_level_keywords": ["employment"], "low_level_keywords": ["Alice", "Acme"]}""");
        }

        // Entity extraction (ingest path): system prompt is the extraction role.
        if (system.Contains("Knowledge Graph Specialist"))
        {
            return Task.FromResult($"""
            entity{D}Alice{D}Person{D}Alice is a software engineer who works at Acme.
            entity{D}Acme{D}Organization{D}Acme is a technology company that employs Alice.
            relation{D}Alice{D}Acme{D}employment{D}Alice works at Acme as a software engineer.
            {C}
            """);
        }

        // Answer generation.
        return Task.FromResult("Alice works at Acme as a software engineer.");
    }

    public async IAsyncEnumerable<string> StreamAsync(string prompt, LlmCompletionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        yield return await CompleteAsync(prompt, options, cancellationToken).ConfigureAwait(false);
    }
}

public class LightRagE2ETests
{
    [Fact]
    public async Task Insert_then_query_end_to_end_with_file_storage()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var rag = FileBasedLightRag.Create(
                dir,
                new FakeLlmModel(),
                FakeEmbedding.Create(),
                new LightRagConfig { EnableLlmCache = true },
                cosineThreshold: 0f);

            await rag.InitializeAsync();
            var processed = await rag.InsertAsync("Alice is a software engineer at Acme. Acme builds software.", "bmw.txt");
            Assert.Equal(1, processed);

            // Files were persisted.
            Assert.True(File.Exists(Path.Combine(dir, "graph_chunk_entity_relation.graphml")));
            Assert.True(File.Exists(Path.Combine(dir, "vdb_entities.json")));
            Assert.True(File.Exists(Path.Combine(dir, "kv_store_full_docs.json")));

            // Query returns the generated answer with retrieved context.
            var result = await rag.QueryAsync("Where does Alice work?", new QueryParam { Mode = QueryMode.Hybrid });
            Assert.Equal("Alice works at Acme as a software engineer.", result.Content);

            // Context-only retrieval surfaces the extracted entities.
            var ctx = await rag.QueryAsync("Where does Alice work?", new QueryParam { Mode = QueryMode.Hybrid, OnlyNeedContext = true });
            Assert.Contains("Alice", ctx.Content);
            Assert.Contains("Acme", ctx.Content);

            await rag.FinalizeAsync();
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task Reingesting_same_document_is_deduplicated()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var rag = FileBasedLightRag.Create(dir, new FakeLlmModel(), FakeEmbedding.Create(),
                cosineThreshold: 0f);
            await rag.InitializeAsync();

            const string doc = "Alice is a software engineer at Acme.";
            Assert.Equal(1, await rag.InsertAsync(doc, "bmw.txt"));
            Assert.Equal(0, await rag.InsertAsync(doc, "bmw.txt")); // dedup: already ingested
        }
        finally { Directory.Delete(dir, true); }
    }

    // ---- A/B: naive (vector-only) vs mix (KG + vector) retrieval ----

    [Fact]
    public async Task Naive_vs_mix_modes_ab_compare_retrieval_context()
    {
        var dir = JsonKvStorageTests.NewTempDir();
        try
        {
            var rag = FileBasedLightRag.Create(dir, new FakeLlmModel(), FakeEmbedding.Create(), cosineThreshold: 0f);
            await rag.InitializeAsync();
            await rag.InsertAsync("Alice is a software engineer at Acme. Acme builds software.", "doc.txt");

            const string question = "Where does Alice work?";
            var naive = await rag.QueryAsync(question, new QueryParam { Mode = QueryMode.Naive, OnlyNeedContext = true });
            var mix = await rag.QueryAsync(question, new QueryParam { Mode = QueryMode.Mix, OnlyNeedContext = true });

            // Both surface the source chunk text (vector retrieval is common to both modes).
            Assert.Contains("Alice", naive.Content);
            Assert.Contains("Alice", mix.Content);

            // A: naive is vector-only — it carries NO knowledge-graph entity/relation sections.
            Assert.DoesNotContain("\"entity\":", naive.Content);
            Assert.DoesNotContain("\"entity1\":", naive.Content);

            // B: mix augments the vector chunks with KG entities AND relations.
            Assert.Contains("\"entity\":", mix.Content);   // entity objects (KG)
            Assert.Contains("\"entity1\":", mix.Content);  // relation objects (KG)
            Assert.Contains("Acme", mix.Content);

            // The mix context is therefore strictly richer than the naive context for the same query.
            var mixLen = (mix.Content ?? "").Length;
            var naiveLen = (naive.Content ?? "").Length;
            Assert.True(mixLen > naiveLen, $"expected mix context (len {mixLen}) to exceed naive (len {naiveLen})");
        }
        finally { Directory.Delete(dir, true); }
    }
}
