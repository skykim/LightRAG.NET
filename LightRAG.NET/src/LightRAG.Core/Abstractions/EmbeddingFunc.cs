namespace LightRAG.Core.Abstractions;

/// <summary>
/// Embedding function wrapper with dimension metadata, ported from
/// <c>EmbeddingFunc</c> in <c>lightrag/utils.py</c>.
/// </summary>
/// <param name="EmbeddingDim">Expected dimension of each embedding vector.</param>
/// <param name="Func">
/// The actual embedding function: given a batch of texts (and optional context),
/// returns one vector per input text.
/// </param>
/// <param name="MaxTokenSize">Optional per-text token limit used for description summarization.</param>
/// <param name="ModelName">Optional model name, used for vector-store data isolation.</param>
/// <param name="SupportsAsymmetric">Whether the underlying function honors the <c>context</c> argument.</param>
public sealed record EmbeddingFunc(
    int EmbeddingDim,
    Func<IReadOnlyList<string>, string?, CancellationToken, Task<float[][]>> Func,
    int? MaxTokenSize = null,
    string? ModelName = null,
    bool SupportsAsymmetric = false)
{
    /// <summary>
    /// Invoke the embedding function and validate that the returned vectors have the
    /// expected dimension and count. <paramref name="context"/> is dropped when the
    /// underlying function does not support asymmetric embedding.
    /// </summary>
    public async Task<float[][]> InvokeAsync(
        IReadOnlyList<string> texts,
        string? context = null,
        CancellationToken cancellationToken = default)
    {
        if (texts.Count == 0)
        {
            return [];
        }

        var effectiveContext = SupportsAsymmetric ? context : null;
        float[][] result = await Func(texts, effectiveContext, cancellationToken).ConfigureAwait(false);

        if (result.Length != texts.Count)
        {
            throw new InvalidOperationException(
                $"Embedding count mismatch: got {result.Length} vectors for {texts.Count} inputs.");
        }

        foreach (var vector in result)
        {
            if (vector.Length != EmbeddingDim)
            {
                throw new InvalidOperationException(
                    $"Embedding dimension mismatch: expected {EmbeddingDim}, got {vector.Length}.");
            }
        }

        return result;
    }
}
