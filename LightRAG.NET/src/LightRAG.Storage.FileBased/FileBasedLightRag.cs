using LightRAG.Core;
using LightRAG.Core.Abstractions;
using LightRAG.Core.Tokenization;

namespace LightRAG.Storage.FileBased;

/// <summary>
/// Convenience factory that assembles a <see cref="LightRag"/> backed entirely by the file-based
/// storages (JSON KV / doc-status, nano-vectordb, GraphML graph). Zero external services.
/// </summary>
public static class FileBasedLightRag
{
    public static LightRag Create(
        string workingDir,
        ILlmModel llm,
        EmbeddingFunc embedding,
        LightRagConfig? config = null,
        ITokenizer? tokenizer = null,
        string workspace = "",
        float cosineThreshold = Core.Configuration.Constants.DefaultCosineThreshold)
    {
        config ??= new LightRagConfig();
        tokenizer ??= new TiktokenTokenizer();

        var storages = new LightRagStorages
        {
            FullDocs = new JsonKvStorage(workingDir, NameSpace.KvStoreFullDocs, workspace),
            TextChunks = new JsonKvStorage(workingDir, NameSpace.KvStoreTextChunks, workspace),
            LlmResponseCache = new JsonKvStorage(workingDir, NameSpace.KvStoreLlmResponseCache, workspace),
            DocStatus = new JsonDocStatusStorage(workingDir, NameSpace.DocStatus, workspace),
            EntitiesVdb = new NanoVectorDbStorage(workingDir, NameSpace.VectorStoreEntities, embedding, cosineThreshold, workspace),
            RelationshipsVdb = new NanoVectorDbStorage(workingDir, NameSpace.VectorStoreRelationships, embedding, cosineThreshold, workspace),
            ChunksVdb = new NanoVectorDbStorage(workingDir, NameSpace.VectorStoreChunks, embedding, cosineThreshold, workspace),
            Graph = new GraphmlGraphStorage(workingDir, embedding, NameSpace.GraphStoreChunkEntityRelation, workspace),
        };

        return new LightRag(config, llm, tokenizer, storages);
    }
}
