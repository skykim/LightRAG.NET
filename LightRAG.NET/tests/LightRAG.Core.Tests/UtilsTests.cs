using LightRAG.Core.Tokenization;
using LightRAG.Core.Utils;

namespace LightRAG.Core.Tests;

public class TokenizerTests
{
    private static readonly TiktokenTokenizer Tokenizer = new("gpt-4o-mini");

    // Golden reference token counts produced by Python tiktoken (encoding_for_model("gpt-4o-mini")).
    [Theory]
    [InlineData("Hello, world!", 4)]
    [InlineData("LightRAG is a knowledge graph RAG system.", 11)]
    [InlineData("The quick brown fox jumps over the lazy dog.", 10)]
    [InlineData("Entity extraction with markers.", 5)]
    public void Matches_python_tiktoken_counts(string text, int expected)
    {
        Assert.Equal(expected, Tokenizer.CountTokens(text));
        Assert.Equal(expected, Tokenizer.Encode(text).Count);
    }

    [Fact]
    public void Encode_then_decode_roundtrips()
    {
        const string text = "The quick brown fox jumps over the lazy dog.";
        var ids = Tokenizer.Encode(text);
        Assert.Equal(text, Tokenizer.Decode(ids));
    }
}

public class HashingTests
{
    [Fact]
    public void ComputeArgsHash_is_md5_hex_of_joined_args()
    {
        // MD5("abc") = 900150983cd24fb0d6963f7d28e17f72
        Assert.Equal("900150983cd24fb0d6963f7d28e17f72", Hashing.ComputeArgsHash("a", "b", "c"));
    }

    [Fact]
    public void ComputeMdHashId_prepends_prefix()
    {
        var id = Hashing.ComputeMdHashId("hello", "chunk-");
        Assert.StartsWith("chunk-", id);
        Assert.Equal(32 + "chunk-".Length, id.Length);
    }
}

public class NormalizeTests
{
    [Fact]
    public void Strips_matching_outer_double_quotes()
    {
        Assert.Equal("Alex", TextUtils.NormalizeExtractedInfo("\"Alex\""));
    }

    [Fact]
    public void Keeps_inner_quotes()
    {
        Assert.Equal("a\"b", TextUtils.NormalizeExtractedInfo("a\"b"));
    }

    [Fact]
    public void Filters_short_numeric_noise()
    {
        Assert.Equal("", TextUtils.NormalizeExtractedInfo("12"));
        Assert.Equal("", TextUtils.NormalizeExtractedInfo("1.2"));
    }

    [Fact]
    public void SplitStringByMultiMarkers_splits_and_trims()
    {
        var parts = TextUtils.SplitStringByMultiMarkers("a<|#|> b <|#|>c", ["<|#|>"]);
        Assert.Equal(["a", "b", "c"], parts);
    }

    [Fact]
    public void RemoveThinkTags_strips_reasoning()
    {
        Assert.Equal("answer", TextUtils.RemoveThinkTags("<think>reasoning here</think>answer"));
    }
}
