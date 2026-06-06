using LightRAG.Core.Abstractions;
using LightRAG.Core.Extraction;
using LightRAG.Core.Llm;
using LightRAG.Core.Prompts;
using LightRAG.Core.Tokenization;

namespace LightRAG.Core.Tests;

/// <summary>An offline ILlmCaller that returns queued canned responses (one per call).</summary>
internal sealed class FakeLlmCaller : ILlmCaller
{
    private readonly Queue<string> _responses;
    public List<(string User, string? System, int HistoryCount)> Calls { get; } = new();

    public FakeLlmCaller(params string[] responses) => _responses = new Queue<string>(responses);

    public Task<string> CompleteAsync(
        string userPrompt, string? systemPrompt = null, IReadOnlyList<ChatMessage>? history = null,
        bool jsonMode = false, string cacheType = "default", string? chunkId = null,
        CancellationToken cancellationToken = default)
    {
        Calls.Add((userPrompt, systemPrompt, history?.Count ?? 0));
        return Task.FromResult(_responses.Count > 0 ? _responses.Dequeue() : "");
    }
}

public class ExtractionTests
{
    private static readonly TiktokenTokenizer Tokenizer = new("gpt-4o-mini");
    private const string D = PromptTemplates.DefaultTupleDelimiter;
    private const string C = PromptTemplates.DefaultCompletionDelimiter;

    [Fact]
    public async Task Parses_text_mode_entities_and_relations()
    {
        var output = $"""
        entity{D}Alice{D}Person{D}Alice is an engineer at Acme.
        entity{D}Acme{D}Organization{D}Acme is a technology company.
        relation{D}Alice{D}Acme{D}employment{D}Alice works at Acme as an engineer.
        {C}
        """;

        var extractor = new EntityExtractor(new FakeLlmCaller(output), Tokenizer,
            new ExtractionOptions { MaxGleaning = 0 });

        var results = await extractor.ExtractAsync([new ChunkInput("chunk-1", "Alice works at Acme.", "doc.txt")]);

        Assert.Single(results);
        var r = results[0];
        Assert.True(r.Nodes.ContainsKey("Alice"));
        Assert.True(r.Nodes.ContainsKey("Acme"));
        Assert.Equal("person", r.Nodes["Alice"][0].EntityType); // lowercased
        Assert.True(r.Edges.ContainsKey(("Alice", "Acme")));
        Assert.Equal("chunk-1", r.Nodes["Alice"][0].SourceId);
    }

    [Fact]
    public async Task Gleaning_adds_new_entities_and_keeps_longer_descriptions()
    {
        var first = $"entity{D}Alice{D}Person{D}Short.\n{C}";
        var glean = $"""
        entity{D}Alice{D}Person{D}Alice is a senior engineer with a long, detailed description.
        entity{D}Bob{D}Person{D}Bob is a manager.
        {C}
        """;

        var extractor = new EntityExtractor(new FakeLlmCaller(first, glean), Tokenizer,
            new ExtractionOptions { MaxGleaning = 1 });

        var results = await extractor.ExtractAsync([new ChunkInput("chunk-1", "text", "doc.txt")]);
        var r = results[0];

        Assert.True(r.Nodes.ContainsKey("Bob")); // new entity from gleaning
        Assert.Contains("senior engineer", r.Nodes["Alice"][0].Description); // longer description won
    }

    [Fact]
    public async Task Recovers_misprefixed_relation_rows()
    {
        // A relation row mistakenly emitted with the "entity" prefix (5 fields) should be recovered.
        var output = $"entity{D}Alice{D}Acme{D}employment{D}Alice works at Acme.\n{C}";
        var extractor = new EntityExtractor(new FakeLlmCaller(output), Tokenizer,
            new ExtractionOptions { MaxGleaning = 0 });

        var results = await extractor.ExtractAsync([new ChunkInput("chunk-1", "t", "doc.txt")]);
        Assert.True(results[0].Edges.ContainsKey(("Alice", "Acme")));
    }
}
