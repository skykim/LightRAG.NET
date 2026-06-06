using System.Runtime.CompilerServices;
using System.Text;
using LightRAG.Core.Abstractions;
using OllamaSharp;
using OllamaSharp.Models;
using OllamaSharp.Models.Chat;

namespace LightRAG.Providers.Ollama;

/// <summary>
/// <see cref="ILlmModel"/> backed by Ollama via OllamaSharp, ported from
/// <c>_ollama_model_if_cache</c> / <c>ollama_model_complete</c> in <c>lightrag/llm/ollama.py</c>.
/// </summary>
public sealed class OllamaLlm : ILlmModel
{
    private readonly OllamaApiClient _client;
    private readonly int? _numCtx;

    public string ModelName { get; }

    /// <param name="model">Ollama chat model (e.g. "qwen2.5:latest").</param>
    /// <param name="host">Ollama host URL.</param>
    /// <param name="numCtx">Optional context window size (Ollama <c>num_ctx</c> option).</param>
    public OllamaLlm(string model, string host = "http://localhost:11434", int? numCtx = null)
    {
        ModelName = model;
        _numCtx = numCtx;
        _client = new OllamaApiClient(new Uri(host), model);
    }

    public async Task<string> CompleteAsync(
        string prompt,
        LlmCompletionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var request = BuildRequest(prompt, options, stream: false);
        var sb = new StringBuilder();
        await foreach (var chunk in _client.ChatAsync(request, cancellationToken).ConfigureAwait(false))
        {
            if (chunk?.Message?.Content is { Length: > 0 } content)
            {
                sb.Append(content);
            }
        }
        return sb.ToString();
    }

    public async IAsyncEnumerable<string> StreamAsync(
        string prompt,
        LlmCompletionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var request = BuildRequest(prompt, options, stream: true);
        await foreach (var chunk in _client.ChatAsync(request, cancellationToken).ConfigureAwait(false))
        {
            if (chunk?.Message?.Content is { Length: > 0 } content)
            {
                yield return content;
            }
        }
    }

    private ChatRequest BuildRequest(string prompt, LlmCompletionOptions? options, bool stream)
    {
        var messages = new List<Message>();
        if (!string.IsNullOrEmpty(options?.SystemPrompt))
        {
            messages.Add(new Message(ChatRole.System, options.SystemPrompt));
        }
        if (options?.History is { Count: > 0 })
        {
            foreach (var message in options.History)
            {
                messages.Add(new Message(MapRole(message.Role), message.Content));
            }
        }
        messages.Add(new Message(ChatRole.User, prompt));

        var request = new ChatRequest
        {
            Model = ModelName,
            Messages = messages,
            Stream = stream,
        };

        if (options?.JsonMode == true)
        {
            request.Format = "json";
        }

        if (_numCtx is not null || options?.Temperature is not null)
        {
            request.Options = new RequestOptions
            {
                NumCtx = _numCtx,
                Temperature = (float?)options?.Temperature,
            };
        }

        return request;
    }

    private static ChatRole MapRole(string role) => role.ToLowerInvariant() switch
    {
        "system" => ChatRole.System,
        "assistant" => ChatRole.Assistant,
        "tool" => ChatRole.Tool,
        _ => ChatRole.User,
    };
}
