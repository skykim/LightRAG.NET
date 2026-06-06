namespace LightRAG.Core.Abstractions;

/// <summary>
/// Tokenizer abstraction, ported from the <c>Tokenizer</c> / <c>TokenizerInterface</c>
/// protocol in <c>lightrag/utils.py</c>. Allows plugging in a custom tokenizer.
/// </summary>
public interface ITokenizer
{
    /// <summary>Encode a string into a list of token ids.</summary>
    IReadOnlyList<int> Encode(string content);

    /// <summary>Decode a list of token ids back into a string.</summary>
    string Decode(IReadOnlyList<int> tokens);

    /// <summary>Count the number of tokens in a string. Defaults to <c>Encode(content).Count</c>.</summary>
    int CountTokens(string content) => Encode(content).Count;
}
