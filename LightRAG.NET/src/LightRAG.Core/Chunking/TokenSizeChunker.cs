using LightRAG.Core.Abstractions;
using LightRAG.Core.Configuration;

namespace LightRAG.Core.Chunking;

/// <summary>A text chunk produced by a chunker. Mirrors the dict shape of <c>TextChunkSchema</c>.</summary>
/// <param name="Tokens">Token count of the chunk.</param>
/// <param name="Content">Chunk text (already trimmed).</param>
/// <param name="ChunkOrderIndex">Zero-based order of the chunk within its document.</param>
public sealed record TextChunk(int Tokens, string Content, int ChunkOrderIndex);

/// <summary>Thrown when a <c>split_by_character_only</c> segment exceeds the token cap.</summary>
public sealed class ChunkTokenLimitExceededException(int chunkTokens, int chunkTokenLimit)
    : Exception($"Chunk exceeds token limit: {chunkTokens} > {chunkTokenLimit}.")
{
    public int ChunkTokens { get; } = chunkTokens;
    public int ChunkTokenLimit { get; } = chunkTokenLimit;
}

/// <summary>
/// Fixed-size token-window chunker — the LightRAG default ("F") strategy. Ported from
/// <c>chunking_by_token_size</c> in <c>lightrag/chunker/token_size.py</c>. The source-span
/// (heading citation) machinery from the Python version is omitted from this core port.
/// </summary>
public static class TokenSizeChunker
{
    public static List<TextChunk> Chunk(
        ITokenizer tokenizer,
        string content,
        int chunkTokenSize = Constants.DefaultChunkSize,
        int chunkOverlapTokenSize = Constants.DefaultChunkOverlapSize,
        string? splitByCharacter = null,
        bool splitByCharacterOnly = false)
    {
        if (chunkTokenSize <= chunkOverlapTokenSize)
        {
            throw new ArgumentException("chunkTokenSize must be greater than chunkOverlapTokenSize.");
        }

        var results = new List<TextChunk>();
        var step = chunkTokenSize - chunkOverlapTokenSize;

        if (!string.IsNullOrEmpty(splitByCharacter))
        {
            var rawChunks = content.Split(splitByCharacter);
            var newChunks = new List<(int Tokens, string Content)>();

            if (splitByCharacterOnly)
            {
                foreach (var chunk in rawChunks)
                {
                    var count = tokenizer.Encode(chunk).Count;
                    if (count > chunkTokenSize)
                    {
                        throw new ChunkTokenLimitExceededException(count, chunkTokenSize);
                    }
                    newChunks.Add((count, chunk));
                }
            }
            else
            {
                foreach (var chunk in rawChunks)
                {
                    var ids = tokenizer.Encode(chunk);
                    if (ids.Count > chunkTokenSize)
                    {
                        for (var start = 0; start < ids.Count; start += step)
                        {
                            var end = Math.Min(start + chunkTokenSize, ids.Count);
                            var slice = ids.Skip(start).Take(end - start).ToList();
                            newChunks.Add((Math.Min(chunkTokenSize, ids.Count - start), tokenizer.Decode(slice)));
                        }
                    }
                    else
                    {
                        newChunks.Add((ids.Count, chunk));
                    }
                }
            }

            for (var i = 0; i < newChunks.Count; i++)
            {
                results.Add(new TextChunk(newChunks[i].Tokens, newChunks[i].Content.Trim(), i));
            }
            return results;
        }

        var tokens = tokenizer.Encode(content);
        var index = 0;
        for (var start = 0; start < Math.Max(tokens.Count, 1); start += step)
        {
            if (start >= tokens.Count)
            {
                break;
            }
            var end = Math.Min(start + chunkTokenSize, tokens.Count);
            var slice = tokens.Skip(start).Take(end - start).ToList();
            var chunkContent = tokenizer.Decode(slice);
            results.Add(new TextChunk(Math.Min(chunkTokenSize, tokens.Count - start), chunkContent.Trim(), index));
            index++;
        }
        return results;
    }
}
