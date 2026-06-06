using System.Text.RegularExpressions;
using LightRAG.Core;
using LightRAG.Core.Abstractions;
using LightRAG.Providers.Ollama;
using LightRAG.Storage.FileBased;

// LightRAG.NET smoke test / demo: ingest a slice of the BMW 5 Series Owner's Handbook into a
// file-based knowledge graph using local Ollama models, then query it. Validates retrieval against
// real sentences from the manual. Exit code 0 = PASS, 1 = FAIL.
//
// Env overrides: LIGHTRAG_LLM_MODEL (default gemma4:e2b), LIGHTRAG_EMBED_MODEL (default
// nomic-embed-text), LIGHTRAG_EMBED_DIM (default 768), OLLAMA_HOST (default http://localhost:11434).

static string Norm(string s) => Regex.Replace(s, @"\s+", " ").Trim();

var manualPath = args.Length > 0
    ? args[0]
    : "/Users/sky.kim/Desktop/Playground/LightRAG/lightrag-unity/Assets/StreamingAssets/Manual/bmw5_manual.txt";

if (!File.Exists(manualPath))
{
    Console.Error.WriteLine($"Manual not found: {manualPath}");
    return 1;
}

// Ingest a focused slice (safety section) so a small local model finishes quickly.
var startLine = args.Length > 1 && int.TryParse(args[1], out var s) ? s : 1000;
var endLine = args.Length > 2 && int.TryParse(args[2], out var e) ? e : 1300;
var allLines = await File.ReadAllLinesAsync(manualPath);
var slice = string.Join("\n", allLines.Skip(startLine - 1).Take(endLine - startLine + 1));

var host = Environment.GetEnvironmentVariable("OLLAMA_HOST") ?? "http://localhost:11434";
var llmModel = Environment.GetEnvironmentVariable("LIGHTRAG_LLM_MODEL") ?? "gemma4:e2b";
var embedModel = Environment.GetEnvironmentVariable("LIGHTRAG_EMBED_MODEL") ?? "nomic-embed-text";
var embedDim = int.TryParse(Environment.GetEnvironmentVariable("LIGHTRAG_EMBED_DIM"), out var d) ? d : 768;

Console.WriteLine("LightRAG.NET BMW smoke test");
Console.WriteLine($"  manual : {manualPath} (lines {startLine}-{endLine}, {slice.Length} chars)");
Console.WriteLine($"  llm    : {llmModel} @ {host}");
Console.WriteLine($"  embed  : {embedModel} (dim {embedDim})");

var workDir = Path.Combine(Path.GetTempPath(), "lightrag-bmw-smoke");
if (Directory.Exists(workDir)) Directory.Delete(workDir, true);
Directory.CreateDirectory(workDir);

var llm = new OllamaLlm(llmModel, host, numCtx: 8192);
var embedding = new OllamaEmbedding(embedModel, host, embeddingDim: embedDim).AsEmbeddingFunc();

var rag = FileBasedLightRag.Create(
    workDir, llm, embedding,
    new LightRagConfig
    {
        // gemma4:e2b is small; temperature 0 keeps extraction stable.
        Temperature = 0,
        MaxAsync = 2,
        LlmTimeout = TimeSpan.FromSeconds(300),
    });

await rag.InitializeAsync();

Console.WriteLine("\n[1/3] Ingesting...");
var sw = System.Diagnostics.Stopwatch.StartNew();
var processed = await rag.InsertAsync(slice, "bmw5_manual.txt");
Console.WriteLine($"      ingested {processed} doc(s) in {sw.Elapsed.TotalSeconds:0.0}s");

// On-disk artifacts created.
var graphFile = Path.Combine(workDir, "graph_chunk_entity_relation.graphml");
Console.WriteLine($"      graph file: {(File.Exists(graphFile) ? "OK" : "MISSING")}");

var failures = new List<string>();

// --- Validation 1: naive (vector) retrieval surfaces a real sentence from the manual. ---
Console.WriteLine("\n[2/3] Naive retrieval validation...");
var naiveCtx = await rag.QueryAsync(
    "What should I do with children or animals when leaving the vehicle?",
    new QueryParam { Mode = QueryMode.Naive, OnlyNeedContext = true });
var naiveNorm = Norm(naiveCtx.Content ?? "");

// These phrases appear verbatim (modulo whitespace) in the ingested slice.
string[] expectedPhrases =
[
    "leave children or animals unsupervised in the vehicle",
    "Apply the parking brake",
    "take the remote control with you and lock the vehicle",
];
var matched = expectedPhrases.Where(p => naiveNorm.Contains(Norm(p), StringComparison.OrdinalIgnoreCase)).ToList();
if (matched.Count > 0)
{
    Console.WriteLine($"      PASS: retrieved context contains real manual sentence(s): {string.Join("; ", matched)}");
}
else
{
    failures.Add("naive retrieval did not surface any expected real sentence");
    Console.WriteLine("      FAIL: expected real sentences not found in retrieved context");
    Console.WriteLine("      context preview: " + naiveNorm[..Math.Min(300, naiveNorm.Length)]);
}

// --- Validation 2: hybrid KG query produces a grounded answer. ---
Console.WriteLine("\n[3/3] Hybrid KG query...");
var answer = await rag.QueryAsync(
    "How do I secure the vehicle against rolling away?",
    new QueryParam { Mode = QueryMode.Hybrid });
Console.WriteLine("      answer: " + Norm(answer.Content ?? ""));
if (string.IsNullOrWhiteSpace(answer.Content) || answer.Content!.Contains("[no-context]"))
{
    failures.Add("hybrid query returned no grounded answer");
    Console.WriteLine("      FAIL: no grounded answer");
}
else
{
    Console.WriteLine("      PASS: answer generated from knowledge graph context");
}

await rag.FinalizeAsync();

Console.WriteLine();
if (failures.Count == 0)
{
    Console.WriteLine("SMOKE TEST PASSED");
    return 0;
}
Console.WriteLine("SMOKE TEST FAILED:");
foreach (var f in failures) Console.WriteLine("  - " + f);
return 1;
