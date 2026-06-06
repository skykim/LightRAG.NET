using LightRAG.Core.Abstractions;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace LightRAG.Core.Utils;

/// <summary>
/// JSON helpers shared by storage backends and the extraction/query engine.
/// Uses Newtonsoft.Json (available both as a NuGet package and as the Unity
/// <c>com.unity.nuget.newtonsoft-json</c> package) so the same assembly works on .NET and in Unity.
/// Converts <see cref="JToken"/> trees into loosely-typed CLR values so the engine treats persisted
/// records like Python dicts.
/// </summary>
public static class JsonHelper
{
    private static readonly JsonSerializerSettings Settings = new()
    {
        // Keep date-like strings as strings; never auto-convert to DateTime.
        DateParseHandling = DateParseHandling.None,
    };

    /// <summary>Serialize a value to JSON. <paramref name="indented"/> controls formatting.</summary>
    public static string Serialize(object? value, bool indented = true)
        => JsonConvert.SerializeObject(value, indented ? Formatting.Indented : Formatting.None, Settings);

    /// <summary>Parse JSON text into a <see cref="JToken"/> (no date coercion).</summary>
    public static JToken Parse(string json)
        => JToken.Parse(json);

    /// <summary>Recursively convert a <see cref="JToken"/> into a native CLR value.</summary>
    public static object? ToClr(JToken token) => token.Type switch
    {
        JTokenType.Object => ToRecord((JObject)token),
        JTokenType.Array => ((JArray)token).Select(ToClr).ToList(),
        JTokenType.Integer => token.Value<long>(),
        JTokenType.Float => token.Value<double>(),
        JTokenType.Boolean => token.Value<bool>(),
        JTokenType.String => token.Value<string>(),
        JTokenType.Null => null,
        JTokenType.Undefined => null,
        _ => token.Value<string>(),
    };

    /// <summary>Convert a JSON object into a <see cref="StorageRecord"/>.</summary>
    public static StorageRecord ToRecord(JObject obj)
    {
        var record = new StorageRecord();
        foreach (var property in obj.Properties())
        {
            record[property.Name] = ToClr(property.Value);
        }
        return record;
    }

    /// <summary>Read a string field from a record, coercing where reasonable. Returns null when absent.</summary>
    public static string? GetString(this IReadOnlyDictionary<string, object?> record, string key)
        => record.TryGetValue(key, out var value) ? value?.ToString() : null;

    /// <summary>Read an integer field from a record. Returns <paramref name="fallback"/> when absent or unparsable.</summary>
    public static int GetInt(this IReadOnlyDictionary<string, object?> record, string key, int fallback = 0)
    {
        if (!record.TryGetValue(key, out var value) || value is null)
        {
            return fallback;
        }
        return value switch
        {
            int i => i,
            long l => (int)l,
            double d => (int)d,
            string s when int.TryParse(s, out var parsed) => parsed,
            _ => fallback,
        };
    }
}
