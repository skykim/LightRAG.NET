using LightRAG.Core.Abstractions;
using LightRAG.Core.Extraction;
using LightRAG.Core.Tokenization;
using LightRAG.Core.Utils;
using LightRAG.Storage.FileBased;

namespace LightRAG.Core.Tests;

public class KnowledgeGraphBuilderTests
{
    private static readonly TiktokenTokenizer Tokenizer = new("gpt-4o-mini");

    private static ChunkExtractionResult MakeResult()
    {
        var r = new ChunkExtractionResult();
        r.Nodes["Alice"] = [new ExtractedNode { EntityName = "Alice", EntityType = "person", Description = "Alice is an engineer.", SourceId = "chunk-1", FilePath = "doc.txt", Timestamp = 1 }];
        r.Nodes["Acme"] = [new ExtractedNode { EntityName = "Acme", EntityType = "organization", Description = "Acme is a company.", SourceId = "chunk-1", FilePath = "doc.txt", Timestamp = 1 }];
        r.Edges[("Alice", "Acme")] = [new ExtractedEdge { SrcId = "Alice", TgtId = "Acme", Weight = 1.0, Description = "Alice works at Acme.", Keywords = "employment", SourceId = "chunk-1", FilePath = "doc.txt", Timestamp = 1 }];
        return r;
    }

    [Fact]
    public async Task Merges_nodes_and_edges_into_graph_and_vdbs()
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

            var builder = new KnowledgeGraphBuilder(graph, entityVdb, relVdb, Tokenizer, new FakeLlmCaller());
            await builder.MergeAsync([MakeResult()]);
            await graph.IndexDoneCallbackAsync();
            await entityVdb.IndexDoneCallbackAsync();
            await relVdb.IndexDoneCallbackAsync();

            // Graph: both nodes and the undirected edge exist.
            Assert.True(await graph.HasNodeAsync("Alice"));
            Assert.True(await graph.HasNodeAsync("Acme"));
            Assert.True(await graph.HasEdgeAsync("Acme", "Alice"));
            var node = await graph.GetNodeAsync("Alice");
            Assert.Equal("person", node!["entity_type"]);

            // Entity VDB keyed by ent-md5(name).
            var aliceId = Hashing.ComputeMdHashId("Alice", "ent-");
            Assert.NotNull(await entityVdb.GetByIdAsync(aliceId));

            // Relationship VDB keyed by rel-md5(sorted src+tgt).
            var relId = Hashing.ComputeMdHashId("Acme" + "Alice", "rel-"); // "Acme" < "Alice" ordinally
            Assert.NotNull(await relVdb.GetByIdAsync(relId));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task Reingest_merges_descriptions_and_sums_weight()
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

            var builder = new KnowledgeGraphBuilder(graph, entityVdb, relVdb, Tokenizer, new FakeLlmCaller());
            await builder.MergeAsync([MakeResult()]);

            // Second ingest from a different chunk with a new description for the same edge.
            var second = new ChunkExtractionResult();
            second.Edges[("Alice", "Acme")] = [new ExtractedEdge { SrcId = "Alice", TgtId = "Acme", Weight = 1.0, Description = "Alice leads engineering at Acme.", Keywords = "leadership", SourceId = "chunk-2", FilePath = "doc2.txt", Timestamp = 2 }];
            await builder.MergeAsync([second]);

            var edge = await graph.GetEdgeAsync("Alice", "Acme");
            Assert.NotNull(edge);
            // Weight summed across the two merges.
            Assert.Equal(2.0, Convert.ToDouble(edge!["weight"]));
            // Keywords union.
            var keywords = edge["keywords"]!.ToString()!;
            Assert.Contains("employment", keywords);
            Assert.Contains("leadership", keywords);
            // Both descriptions retained (joined; <8 fragments -> no LLM summary).
            var desc = edge["description"]!.ToString()!;
            Assert.Contains("works at", desc);
            Assert.Contains("leads engineering", desc);
        }
        finally { Directory.Delete(dir, true); }
    }
}
