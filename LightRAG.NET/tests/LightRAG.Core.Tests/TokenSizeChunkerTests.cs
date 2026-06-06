using LightRAG.Core.Chunking;
using LightRAG.Core.Tokenization;

namespace LightRAG.Core.Tests;

public class TokenSizeChunkerTests
{
    private static readonly TiktokenTokenizer Tokenizer = new("gpt-4o-mini");

    [Fact]
    public void Matches_python_chunk_boundaries()
    {
        // Reference produced by the Python token_size chunker (tiktoken gpt-4o-mini):
        // 101 tokens -> 7 chunks, first=20 tokens, last=11 tokens (size=20, overlap=5).
        var content = string.Concat(Enumerable.Repeat("The quick brown fox jumps over the lazy dog. ", 10));

        var chunks = TokenSizeChunker.Chunk(Tokenizer, content, chunkTokenSize: 20, chunkOverlapTokenSize: 5);

        Assert.Equal(7, chunks.Count);
        Assert.Equal(20, chunks[0].Tokens);
        Assert.Equal(11, chunks[^1].Tokens);
        // Order indices are sequential.
        Assert.Equal(Enumerable.Range(0, 7), chunks.Select(c => c.ChunkOrderIndex));
    }

    [Fact]
    public void Single_chunk_when_under_limit()
    {
        var chunks = TokenSizeChunker.Chunk(Tokenizer, "short text", chunkTokenSize: 1200, chunkOverlapTokenSize: 100);
        Assert.Single(chunks);
        Assert.Equal("short text", chunks[0].Content);
    }

    [Fact]
    public void Empty_content_yields_no_chunks()
    {
        var chunks = TokenSizeChunker.Chunk(Tokenizer, "", chunkTokenSize: 1200, chunkOverlapTokenSize: 100);
        Assert.Empty(chunks);
    }
}
