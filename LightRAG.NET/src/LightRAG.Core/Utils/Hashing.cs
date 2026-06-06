using System.Security.Cryptography;
using System.Text;

namespace LightRAG.Core.Utils;

/// <summary>
/// MD5-based content identifiers, ported from <c>compute_args_hash</c> / <c>compute_mdhash_id</c>
/// in <c>lightrag/utils.py</c>. Written to the netstandard2.1 API surface so the assembly loads in Unity.
/// </summary>
public static class Hashing
{
    /// <summary>Compute the MD5 hex digest of the concatenation of the given arguments.</summary>
    public static string ComputeArgsHash(params string[] args)
    {
        var joined = string.Concat(args);
        // UTF-8 encoding mirrors Python's str.encode("utf-8"); invalid sequences are replaced.
        byte[] bytes = Encoding.UTF8.GetBytes(joined);
        using var md5 = MD5.Create();
        byte[] hash = md5.ComputeHash(bytes);
        var sb = new StringBuilder(hash.Length * 2);
        foreach (var b in hash)
        {
            sb.Append(b.ToString("x2"));
        }
        return sb.ToString();
    }

    /// <summary>Compute a unique id for content: <paramref name="prefix"/> + MD5(content).</summary>
    public static string ComputeMdHashId(string content, string prefix = "")
        => prefix + ComputeArgsHash(content);
}
