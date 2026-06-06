using LightRAG.Core.Abstractions;
using OllamaSharp;
using OllamaSharp.Models;

namespace LightRAG.Providers.Ollama;

/// <summary>
/// Ollama embedding provider, ported from <c>ollama_embed</c> in <c>lightrag/llm/ollama.py</c>.
/// Defaults to the <c>bge-m3:latest</c> model (1024-dim, asymmetric query/document support).
/// Use <see cref="AsEmbeddingFunc"/> to obtain the <see cref="EmbeddingFunc"/> the engine consumes.
/// </summary>
public sealed class OllamaEmbedding
{
    private readonly OllamaApiClient _client;

    public string EmbedModel { get; }
    public int EmbeddingDim { get; }
    public int MaxTokenSize { get; }
    public string? QueryPrefix { get; }
    public string? DocumentPrefix { get; }

    public OllamaEmbedding(
        string embedModel = "bge-m3:latest",
        string host = "http://localhost:11434",
        int embeddingDim = 1024,
        int maxTokenSize = 8192,
        string? queryPrefix = null,
        string? documentPrefix = null)
    {
        EmbedModel = embedModel;
        EmbeddingDim = embeddingDim;
        MaxTokenSize = maxTokenSize;
        QueryPrefix = queryPrefix;
        DocumentPrefix = documentPrefix;
        _client = new OllamaApiClient(new Uri(host), embedModel);
    }

    public EmbeddingFunc AsEmbeddingFunc() => new(
        EmbeddingDim: EmbeddingDim,
        Func: EmbedAsync,
        MaxTokenSize: MaxTokenSize,
        ModelName: EmbedModel,
        SupportsAsymmetric: true);

    private async Task<float[][]> EmbedAsync(IReadOnlyList<string> texts, string? context, CancellationToken cancellationToken)
    {
        // Apply context-based prefixes (asymmetric embedding), mirroring ollama_embed.
        var inputs = texts.ToList();
        if (context == "query" && QueryPrefix is not null)
        {
            inputs = inputs.Select(t => QueryPrefix + t).ToList();
        }
        else if (context == "document" && DocumentPrefix is not null)
        {
            inputs = inputs.Select(t => DocumentPrefix + t).ToList();
        }

        var request = new EmbedRequest { Model = EmbedModel, Input = inputs };
        var response = await _client.EmbedAsync(request, cancellationToken).ConfigureAwait(false);
        return response.Embeddings.Select(v => v.ToArray()).ToArray();
    }
}
