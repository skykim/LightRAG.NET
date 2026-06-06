using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace LightRAG.Core.Utils;

/// <summary>
/// Lenient JSON recovery for LLM output, replacing the Python <c>json_repair</c> dependency.
/// Strategy: locate the JSON body, strip code fences, remove trailing commas, then parse.
/// Returns null when the text cannot be recovered into valid JSON.
/// </summary>
public static class JsonRepair
{
    private static readonly JsonLoadSettings LoadSettings = new();

    /// <summary>Attempt to parse possibly-malformed JSON. Returns a parsed <see cref="JToken"/> or null.</summary>
    public static JToken? TryParse(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }

        foreach (var candidate in Candidates(text!))
        {
            try
            {
                using var reader = new JsonTextReader(new System.IO.StringReader(candidate))
                {
                    DateParseHandling = DateParseHandling.None,
                };
                return JToken.Load(reader, LoadSettings);
            }
            catch (JsonException)
            {
                // try the next repair candidate
            }
        }
        return null;
    }

    private static IEnumerable<string> Candidates(string text)
    {
        yield return text;

        var stripped = StripCodeFences(text).Trim();
        if (stripped != text)
        {
            yield return stripped;
        }

        var body = ExtractJsonBody(stripped);
        if (body is not null)
        {
            yield return body;

            var noTrailingCommas = RemoveTrailingCommas(body);
            if (noTrailingCommas != body)
            {
                yield return noTrailingCommas;
            }
        }
    }

    private static string StripCodeFences(string text)
    {
        var t = text.Trim();
        if (t.StartsWith("```"))
        {
            var firstNewline = t.IndexOf('\n');
            if (firstNewline >= 0)
            {
                t = t[(firstNewline + 1)..];
            }
            if (t.EndsWith("```"))
            {
                t = t[..^3];
            }
        }
        return t;
    }

    private static string? ExtractJsonBody(string text)
    {
        var firstObj = text.IndexOf('{');
        var firstArr = text.IndexOf('[');
        var start = (firstObj, firstArr) switch
        {
            (< 0, < 0) => -1,
            (>= 0, < 0) => firstObj,
            (< 0, >= 0) => firstArr,
            _ => Math.Min(firstObj, firstArr),
        };
        if (start < 0)
        {
            return null;
        }

        var open = text[start];
        var close = open == '{' ? '}' : ']';
        var lastClose = text.LastIndexOf(close);
        if (lastClose <= start)
        {
            return null;
        }
        return text[start..(lastClose + 1)];
    }

    private static string RemoveTrailingCommas(string json)
    {
        var sb = new StringBuilder(json.Length);
        for (var i = 0; i < json.Length; i++)
        {
            if (json[i] == ',')
            {
                var j = i + 1;
                while (j < json.Length && char.IsWhiteSpace(json[j]))
                {
                    j++;
                }
                if (j < json.Length && (json[j] == '}' || json[j] == ']'))
                {
                    continue; // skip the trailing comma
                }
            }
            sb.Append(json[i]);
        }
        return sb.ToString();
    }
}
