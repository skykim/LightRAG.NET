using System.Diagnostics;
using System.Text.RegularExpressions;
using LightRAG.Core;
using LightRAG.Core.Abstractions;
using LightRAG.Core.Chunking;
using LightRAG.Core.Configuration;
using LightRAG.Core.Pipeline;
using LightRAG.Core.Tokenization;
using LightRAG.Core.Utils;
using LightRAG.Providers.Ollama;
using LightRAG.Storage.FileBased;

// LightRAG.NET retrieval + speed benchmark on the BMW manual.
//   Phase 1: vector-only ingest of the (whole) manual, timed.
//   Phase 2: N retrieval queries (sentence -> source chunk), measuring recall@k, MRR, and latency.
//   Phase 3 (optional --kg): full KG ingest of a slice, timed, plus a few hybrid queries timed.
//
// Args: [manualPath] [maxLines=all] [numQueries=1000] [topK=10] [--kg]
// Env : OLLAMA_HOST, LIGHTRAG_LLM_MODEL (gemma4:e2b), LIGHTRAG_EMBED_MODEL (nomic-embed-text), LIGHTRAG_EMBED_DIM (768)

static string Norm(string s) => Regex.Replace(s ?? "", @"\s+", " ").Trim();

var positional = args.Where(a => !a.StartsWith("--")).ToArray();
var manualPath = positional.FirstOrDefault()
    ?? "/Users/sky.kim/Desktop/Playground/LightRAG/lightrag-unity/Assets/StreamingAssets/Manual/bmw5_manual.txt";
var maxLines = positional.Length > 1 && int.TryParse(positional[1], out var ml) ? ml : 0; // 0 = all
var numQueries = positional.Length > 2 && int.TryParse(positional[2], out var nq) ? nq : 1000;
var topK = positional.Length > 3 && int.TryParse(positional[3], out var tk) ? tk : 10;
var doKg = args.Contains("--kg");
var dumpQueries = args.FirstOrDefault(a => a.StartsWith("--dump-queries="))?["--dump-queries=".Length..];
var loadQueries = args.FirstOrDefault(a => a.StartsWith("--load-queries="))?["--load-queries=".Length..];

if (!File.Exists(manualPath)) { Console.Error.WriteLine($"Manual not found: {manualPath}"); return 1; }

var host = Environment.GetEnvironmentVariable("OLLAMA_HOST") ?? "http://localhost:11434";
var llmModel = Environment.GetEnvironmentVariable("LIGHTRAG_LLM_MODEL") ?? "gemma4:e2b";
var embedModel = Environment.GetEnvironmentVariable("LIGHTRAG_EMBED_MODEL") ?? "nomic-embed-text";
var embedDim = int.TryParse(Environment.GetEnvironmentVariable("LIGHTRAG_EMBED_DIM"), out var d) ? d : 768;

var allLines = await File.ReadAllLinesAsync(manualPath);
var usedLines = maxLines > 0 ? allLines.Take(maxLines).ToArray() : allLines;
var text = string.Join("\n", usedLines);
var tokenizer = new TiktokenTokenizer("gpt-4o-mini");
var totalTokens = tokenizer.CountTokens(text);

Console.WriteLine("================ LightRAG.NET Retrieval Benchmark ================");
Console.WriteLine($"manual    : {manualPath}");
Console.WriteLine($"corpus    : {usedLines.Length:N0} lines, {text.Length:N0} chars, {totalTokens:N0} tokens");
Console.WriteLine($"models    : llm={llmModel}  embed={embedModel} (dim {embedDim}) @ {host}");
Console.WriteLine($"params    : numQueries={numQueries}  topK={topK}  kgPhase={doKg}");

// Deterministic chunk enumeration (same params the pipeline uses) -> gold (id, content) map.
var chunks = TokenSizeChunker.Chunk(tokenizer, text, Constants.DefaultChunkSize, Constants.DefaultChunkOverlapSize);
var chunkContents = chunks.Select(c => c.Content).ToList();
var chunkIds = chunkContents.Select(c => Hashing.ComputeMdHashId(c, "chunk-")).ToList();
Console.WriteLine($"chunks    : {chunks.Count:N0} (size={Constants.DefaultChunkSize}, overlap={Constants.DefaultChunkOverlapSize})");

var llm = new OllamaLlm(llmModel, host, numCtx: 8192);
var embedding = new OllamaEmbedding(embedModel, host, embeddingDim: embedDim).AsEmbeddingFunc();

var workDir = Path.Combine(Path.GetTempPath(), "lightrag-benchmark");
if (Directory.Exists(workDir)) Directory.Delete(workDir, true);
Directory.CreateDirectory(workDir);

// ---------------- Phase 1: vector-only ingest ----------------
Console.WriteLine("\n[Phase 1] Vector-only ingest (chunk + embed, no KG extraction)...");
var rag = FileBasedLightRag.Create(workDir, llm, embedding,
    new LightRagConfig
    {
        Pipeline = new PipelineOptions { SkipEntityExtraction = true },
        EnableLlmCache = false,
    }, tokenizer: tokenizer, cosineThreshold: -1f); // -1 keeps all candidates for ranking
await rag.InitializeAsync();

var ingestSw = Stopwatch.StartNew();
await rag.InsertAsync(text, "bmw5_manual.txt");
ingestSw.Stop();
var ingestSec = ingestSw.Elapsed.TotalSeconds;
Console.WriteLine($"  ingest time      : {ingestSec:0.0}s");
Console.WriteLine($"  throughput       : {chunks.Count / ingestSec:0.0} chunks/s, {totalTokens / ingestSec:N0} tokens/s");

// ---------------- Build query set: sentence -> gold chunk(s) ----------------
var rng = new Random(42);
var sentenceSplit = new Regex(@"(?<=[.!?])\s+");
var candidates = new List<(string Query, int ChunkIndex)>();
for (var i = 0; i < chunkContents.Count; i++)
{
    foreach (var raw in sentenceSplit.Split(chunkContents[i]))
    {
        var s = Norm(raw);
        if (s.Length is >= 40 and <= 300 && Regex.IsMatch(s, "[A-Za-z]{4,}"))
        {
            candidates.Add((s, i));
        }
    }
}
List<(string Query, int ChunkIndex)> queries;
if (loadQueries is not null)
{
    // Use an externally supplied query set (for 1:1 cross-implementation comparison).
    queries = File.ReadAllLines(loadQueries).Where(l => l.Trim().Length > 0).Select(l => (l, -1)).ToList();
    Console.WriteLine($"\n[Phase 2] Retrieval: {queries.Count:N0} queries (loaded from {loadQueries})...");
}
else
{
    candidates = candidates.GroupBy(c => c.Query).Select(g => g.First()).OrderBy(_ => rng.Next()).ToList();
    queries = candidates.Take(numQueries).ToList();
    Console.WriteLine($"\n[Phase 2] Retrieval: {queries.Count:N0} queries (from {candidates.Count:N0} candidate sentences)...");
    if (queries.Count < numQueries)
    {
        Console.WriteLine($"  note: only {queries.Count} distinct qualifying sentences available (< {numQueries}).");
    }
}
if (dumpQueries is not null)
{
    File.WriteAllLines(dumpQueries, queries.Select(q => q.Query));
    Console.WriteLine($"  dumped {queries.Count} queries to {dumpQueries}");
}

await rag.RetrieveChunksAsync("warmup query", topK); // warm up embedding endpoint

int hit1 = 0, hit5 = 0, hitK = 0;
double mrrSum = 0;
var latencies = new List<double>(queries.Count);
var searchSw = Stopwatch.StartNew();
for (var q = 0; q < queries.Count; q++)
{
    var (query, srcIdx) = queries[q];
    var goldIds = new HashSet<string>();
    for (var i = 0; i < chunkContents.Count; i++)
    {
        if (i == srcIdx || chunkContents[i].Contains(query)) goldIds.Add(chunkIds[i]);
    }

    var sw = Stopwatch.StartNew();
    var results = await rag.RetrieveChunksAsync(query, topK);
    sw.Stop();
    latencies.Add(sw.Elapsed.TotalMilliseconds);

    var rank = -1;
    for (var r = 0; r < results.Count; r++)
    {
        if (results[r].GetString("id") is { } id && goldIds.Contains(id)) { rank = r; break; }
    }
    if (rank >= 0)
    {
        if (rank == 0) hit1++;
        if (rank < 5) hit5++;
        hitK++;
        mrrSum += 1.0 / (rank + 1);
    }
    if ((q + 1) % 200 == 0) Console.WriteLine($"  ...{q + 1}/{queries.Count} done");
}
searchSw.Stop();

latencies.Sort();
double Pct(double p) => latencies[Math.Min(latencies.Count - 1, (int)(p * latencies.Count))];
var n = queries.Count;

Console.WriteLine($"\n---------------- Retrieval quality (n={n}) ----------------");
Console.WriteLine($"  Recall@1   : {hit1 * 100.0 / n:0.0}%  ({hit1}/{n})");
Console.WriteLine($"  Recall@5   : {hit5 * 100.0 / n:0.0}%  ({hit5}/{n})");
Console.WriteLine($"  Recall@{topK,-2} : {hitK * 100.0 / n:0.0}%  ({hitK}/{n})");
Console.WriteLine($"  MRR@{topK}    : {mrrSum / n:0.000}");
Console.WriteLine($"\n---------------- Search speed ----------------");
Console.WriteLine($"  total search time: {searchSw.Elapsed.TotalSeconds:0.0}s for {n} queries");
Console.WriteLine($"  per query  avg   : {latencies.Average():0.0} ms");
Console.WriteLine($"  per query  p50   : {Pct(0.50):0.0} ms");
Console.WriteLine($"  per query  p95   : {Pct(0.95):0.0} ms");
Console.WriteLine($"  per query  min/max: {latencies.First():0.0} / {latencies.Last():0.0} ms");
Console.WriteLine($"  throughput       : {n / searchSw.Elapsed.TotalSeconds:0.0} queries/s");

await rag.FinalizeAsync();

// ---------------- Phase 3 (optional): KG ingest + hybrid query timing ----------------
if (doKg)
{
    var kgLines = string.Join("\n", usedLines.Skip(999).Take(601)); // safety section, ~600 lines
    var kgChunks = TokenSizeChunker.Chunk(tokenizer, kgLines, Constants.DefaultChunkSize, Constants.DefaultChunkOverlapSize);
    Console.WriteLine($"\n[Phase 3] KG ingest (full extraction) of a {kgChunks.Count}-chunk slice...");
    var kgDir = Path.Combine(Path.GetTempPath(), "lightrag-benchmark-kg");
    if (Directory.Exists(kgDir)) Directory.Delete(kgDir, true);
    Directory.CreateDirectory(kgDir);

    var kgRag = FileBasedLightRag.Create(kgDir, llm, embedding,
        new LightRagConfig { Temperature = 0, MaxAsync = 2, LlmTimeout = TimeSpan.FromSeconds(300) },
        tokenizer: tokenizer);
    await kgRag.InitializeAsync();
    var kgSw = Stopwatch.StartNew();
    await kgRag.InsertAsync(kgLines, "bmw5_manual.txt");
    kgSw.Stop();
    Console.WriteLine($"  KG ingest time   : {kgSw.Elapsed.TotalSeconds:0.0}s for {kgChunks.Count} chunks " +
                      $"({kgSw.Elapsed.TotalSeconds / Math.Max(1, kgChunks.Count):0.0}s/chunk)");

    string[] kgQueries =
    {
        "How do I secure the vehicle against rolling away?",
        "What should I do with children when leaving the vehicle?",
        "When does the vehicle switch to idle state?",
    };
    var kgLat = new List<double>();
    foreach (var qq in kgQueries)
    {
        var sw = Stopwatch.StartNew();
        var res = await kgRag.QueryAsync(qq, new QueryParam { Mode = QueryMode.Hybrid });
        sw.Stop();
        kgLat.Add(sw.Elapsed.TotalSeconds);
        var grounded = !string.IsNullOrWhiteSpace(res.Content) && !res.Content!.Contains("[no-context]");
        Console.WriteLine($"  hybrid query     : {sw.Elapsed.TotalSeconds:0.0}s  grounded={grounded}");
    }
    Console.WriteLine($"  hybrid avg       : {kgLat.Average():0.0}s/query (incl. LLM keyword extraction + answer generation)");
    await kgRag.FinalizeAsync();
}

Console.WriteLine("\n================ Benchmark complete ================");
return 0;
