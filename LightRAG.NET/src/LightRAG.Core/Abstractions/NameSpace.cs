namespace LightRAG.Core.Abstractions;

/// <summary>
/// Canonical storage namespace identifiers, ported from <c>NameSpace</c> in <c>lightrag/namespace.py</c>.
/// These names must not change — they are part of the on-disk layout.
/// </summary>
public static class NameSpace
{
    public const string KvStoreFullDocs = "full_docs";
    public const string KvStoreTextChunks = "text_chunks";
    public const string KvStoreLlmResponseCache = "llm_response_cache";
    public const string KvStoreFullEntities = "full_entities";
    public const string KvStoreFullRelations = "full_relations";

    public const string VectorStoreEntities = "entities";
    public const string VectorStoreRelationships = "relationships";
    public const string VectorStoreChunks = "chunks";

    public const string GraphStoreChunkEntityRelation = "chunk_entity_relation";

    public const string DocStatus = "doc_status";
}
