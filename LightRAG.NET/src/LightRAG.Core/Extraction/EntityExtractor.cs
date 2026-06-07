using LightRAG.Core.Abstractions;
using LightRAG.Core.Configuration;
using LightRAG.Core.Llm;
using LightRAG.Core.Prompts;

namespace LightRAG.Core.Extraction;

/// <summary>A chunk to extract from.</summary>
/// <param name="Key">Chunk id (e.g. "chunk-abc").</param>
/// <param name="Content">Chunk text.</param>
/// <param name="FilePath">Source file path / basename for citations.</param>
public sealed record ChunkInput(string Key, string Content, string FilePath);

/// <summary>Configuration for entity/relationship extraction.</summary>
public sealed record ExtractionOptions
{
    public string Language { get; init; } = Constants.DefaultSummaryLanguage;
    public string EntityTypesGuidance { get; init; } = PromptTemplates.DefaultEntityTypesGuidance;
    public IReadOnlyList<string> TextExamples { get; init; } = PromptTemplates.EntityExtractionExamples;
    public IReadOnlyList<string> JsonExamples { get; init; } = PromptTemplates.EntityExtractionJsonExamples;
    public int MaxTotalRecords { get; init; } = Constants.DefaultMaxExtractionRecords;
    public int MaxEntityRecords { get; init; } = Constants.DefaultMaxExtractionEntities;
    public int MaxGleaning { get; init; } = Constants.DefaultMaxGleaning;
    public int MaxExtractInputTokens { get; init; } = Constants.DefaultMaxExtractInputTokens;
    public bool UseJson { get; init; }
}

/// <summary>
/// Per-chunk entity/relationship extraction with a gleaning pass, ported from
/// <c>extract_entities</c> / <c>_process_single_content</c> in <c>lightrag/operate.py</c>.
/// Chunk-level concurrency is delegated to the scheduler inside <see cref="ILlmCaller"/>.
/// </summary>
public sealed class EntityExtractor
{
    private readonly ILlmCaller _llm;
    private readonly ITokenizer _tokenizer;
    private readonly ExtractionOptions _options;

    public EntityExtractor(ILlmCaller llm, ITokenizer tokenizer, ExtractionOptions? options = null)
    {
        _llm = llm;
        _tokenizer = tokenizer;
        _options = options ?? new ExtractionOptions();
    }

    /// <summary>Extract from all chunks concurrently; returns one result per chunk.</summary>
    public async Task<IReadOnlyList<ChunkExtractionResult>> ExtractAsync(
        IReadOnlyList<ChunkInput> chunks,
        CancellationToken cancellationToken = default)
    {
        var systemPrompt = BuildSystemPrompt();
        var tasks = chunks.Select(chunk => ProcessChunkAsync(chunk, systemPrompt, cancellationToken));
        var results = await Task.WhenAll(tasks).ConfigureAwait(false);
        return results;
    }

    private async Task<ChunkExtractionResult> ProcessChunkAsync(
        ChunkInput chunk,
        string systemPrompt,
        CancellationToken cancellationToken)
    {
        var userTemplate = _options.UseJson
            ? PromptTemplates.EntityExtractionJsonUserPrompt
            : PromptTemplates.EntityExtractionUserPrompt;
        var userPrompt = BuildUserPrompt(userTemplate, chunk.Content);

        var firstResult = await _llm.CompleteAsync(
            userPrompt, systemPrompt, history: null, jsonMode: _options.UseJson,
            cacheType: "extract", chunkId: chunk.Key, cancellationToken).ConfigureAwait(false);

        var result = Parse(firstResult, chunk);

        // Gleaning: one corrective pass replaying the first turn.
        if (_options.MaxGleaning > 0)
        {
            var continuePrompt = _options.UseJson
                ? PromptTemplates.EntityContinueExtractionJsonUserPrompt
                : BuildUserPrompt(PromptTemplates.EntityContinueExtractionUserPrompt, chunk.Content);
            continuePrompt = _options.UseJson ? RenderContextOnly(continuePrompt) : continuePrompt;

            var history = new List<ChatMessage>
            {
                new("user", userPrompt),
                new("assistant", firstResult),
            };

            if (ShouldRunGleaning(systemPrompt, history, continuePrompt))
            {
                var gleanResult = await _llm.CompleteAsync(
                    continuePrompt, systemPrompt, history, jsonMode: _options.UseJson,
                    cacheType: "extract", chunkId: chunk.Key, cancellationToken).ConfigureAwait(false);

                var gleaned = Parse(gleanResult, chunk);
                MergeKeepingLongerDescriptions(result, gleaned);
            }
        }

        return result;
    }

    private bool ShouldRunGleaning(string systemPrompt, List<ChatMessage> history, string continuePrompt)
    {
        if (_options.MaxExtractInputTokens <= 0)
        {
            return true; // guard disabled
        }
        var tokenCount = _tokenizer.CountTokens(systemPrompt)
            + history.Sum(m => _tokenizer.CountTokens(m.Content))
            + _tokenizer.CountTokens(continuePrompt);
        return tokenCount <= _options.MaxExtractInputTokens;
    }

    private ChunkExtractionResult Parse(string llmOutput, ChunkInput chunk)
    {
        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        return _options.UseJson
            ? ExtractionParser.ParseJsonMode(llmOutput, chunk.Key, timestamp, chunk.FilePath)
            : ExtractionParser.ParseTextMode(llmOutput, chunk.Key, timestamp, chunk.FilePath);
    }

    private static void MergeKeepingLongerDescriptions(ChunkExtractionResult into, ChunkExtractionResult from)
    {
        foreach (var (name, gleanNodes) in from.Nodes)
        {
            if (into.Nodes.TryGetValue(name, out var existing))
            {
                var existingLen = existing.Count > 0 ? existing[0].Description.Length : 0;
                var gleanLen = gleanNodes.Count > 0 ? gleanNodes[0].Description.Length : 0;
                if (gleanLen > existingLen)
                {
                    into.Nodes[name] = gleanNodes;
                }
            }
            else
            {
                into.Nodes[name] = gleanNodes;
            }
        }

        foreach (var (key, gleanEdges) in from.Edges)
        {
            if (into.Edges.TryGetValue(key, out var existing))
            {
                var existingLen = existing.Count > 0 ? existing[0].Description.Length : 0;
                var gleanLen = gleanEdges.Count > 0 ? gleanEdges[0].Description.Length : 0;
                if (gleanLen > existingLen)
                {
                    into.Edges[key] = gleanEdges;
                }
            }
            else
            {
                into.Edges[key] = gleanEdges;
            }
        }
    }

    // ---- prompt construction ----

    private string BuildSystemPrompt()
    {
        if (_options.UseJson)
        {
            var examples = string.Join("\n", _options.JsonExamples);
            return PromptRenderer.Render(PromptTemplates.EntityExtractionJsonSystemPrompt, ContextBase(examples));
        }
        else
        {
            // Render examples first (substitute delimiters), then inject into the system prompt.
            var joined = string.Join("\n", _options.TextExamples);
            var examples = PromptRenderer.Render(joined,
                ("tuple_delimiter", PromptTemplates.DefaultTupleDelimiter),
                ("completion_delimiter", PromptTemplates.DefaultCompletionDelimiter),
                ("entity_types_guidance", _options.EntityTypesGuidance),
                ("language", _options.Language));
            return PromptRenderer.Render(PromptTemplates.EntityExtractionSystemPrompt, ContextBase(examples));
        }
    }

    private Dictionary<string, string?> ContextBase(string examples) => new()
    {
        ["tuple_delimiter"] = PromptTemplates.DefaultTupleDelimiter,
        ["completion_delimiter"] = PromptTemplates.DefaultCompletionDelimiter,
        ["entity_types_guidance"] = _options.EntityTypesGuidance,
        ["examples"] = examples,
        ["language"] = _options.Language,
        ["max_total_records"] = _options.MaxTotalRecords.ToString(),
        ["max_entity_records"] = _options.MaxEntityRecords.ToString(),
    };

    private string BuildUserPrompt(string template, string content)
    {
        var ctx = ContextBase(string.Empty);
        ctx["input_text"] = content;
        return PromptRenderer.Render(template, ctx);
    }

    private string RenderContextOnly(string template)
        => PromptRenderer.Render(template, ContextBase(string.Empty));
}
