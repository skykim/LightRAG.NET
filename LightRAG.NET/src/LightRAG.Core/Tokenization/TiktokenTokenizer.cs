using LightRAG.Core.Abstractions;
using MlTokenizer = Microsoft.ML.Tokenizers.TiktokenTokenizer;

namespace LightRAG.Core.Tokenization;

/// <summary>
/// <see cref="ITokenizer"/> backed by <c>Microsoft.ML.Tokenizers</c>' tiktoken implementation,
/// ported from <c>TiktokenTokenizer</c> in <c>lightrag/utils.py</c>.
/// Defaults to the <c>gpt-4o-mini</c> model (o200k_base encoding) to match the Python default.
/// </summary>
public sealed class TiktokenTokenizer : ITokenizer
{
    private readonly MlTokenizer _tokenizer;

    public string ModelName { get; }

    public TiktokenTokenizer(string modelName = "gpt-4o-mini")
    {
        ModelName = modelName;
        // CreateForModel resolves the encoding (e.g. o200k_base / cl100k_base) and loads
        // the vocab from the referenced data package.
        _tokenizer = MlTokenizer.CreateForModel(modelName);
    }

    public IReadOnlyList<int> Encode(string content) => _tokenizer.EncodeToIds(content);

    public string Decode(IReadOnlyList<int> tokens) => _tokenizer.Decode(tokens);

    public int CountTokens(string content) => _tokenizer.CountTokens(content);
}
