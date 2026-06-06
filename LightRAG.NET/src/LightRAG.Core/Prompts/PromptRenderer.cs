using System.Text;

namespace LightRAG.Core.Prompts;

/// <summary>
/// Renders prompt templates using named placeholders, replicating the relevant subset of
/// Python's <c>str.format(**kwargs)</c> semantics used by <c>lightrag/prompt.py</c>:
/// <list type="bullet">
///   <item>Each <c>{key}</c> for a supplied key is replaced with its value.</item>
///   <item>Doubled braces <c>{{</c> / <c>}}</c> are unescaped to <c>{</c> / <c>}</c>.</item>
///   <item>Unmatched single braces (e.g. literal JSON inside inserted examples) are left intact.</item>
/// </list>
/// This avoids the brittleness of a full format parser while matching how LightRAG inserts
/// pre-rendered example blocks (which contain raw JSON braces) into the <c>{examples}</c> slot.
/// </summary>
public static class PromptRenderer
{
    public static string Render(string template, IReadOnlyDictionary<string, string?> values)
    {
        var sb = new StringBuilder(template);
        foreach (var (key, value) in values)
        {
            sb.Replace("{" + key + "}", value ?? string.Empty);
        }

        // Unescape literal braces last, mirroring str.format's single-pass behavior.
        sb.Replace("{{", "{");
        sb.Replace("}}", "}");
        return sb.ToString();
    }

    /// <summary>Convenience overload accepting inline (key, value) tuples.</summary>
    public static string Render(string template, params (string Key, string? Value)[] values)
    {
        var map = new Dictionary<string, string?>(values.Length);
        foreach (var (key, value) in values)
        {
            map[key] = value;
        }
        return Render(template, map);
    }
}
