using System.Text;
using System.Text.RegularExpressions;
using LightRAG.Core.Abstractions;

namespace LightRAG.Core.Utils;

/// <summary>
/// Text helpers ported from <c>lightrag/utils.py</c>: marker splitting, token-size truncation,
/// think-tag removal, and entity/relation name normalization.
/// Uses <see cref="Regex"/> instances (not the net7+ <c>[GeneratedRegex]</c> source generator) so the
/// assembly targets netstandard2.1 and loads in Unity.
/// </summary>
public static class TextUtils
{
    private static readonly Regex FloatRegex = new(@"^[-+]?[0-9]*\.?[0-9]+$", RegexOptions.Compiled);
    private static readonly Regex OrphanThinkPrefixRegex = new(@"^((?!<think>).)*?</think>", RegexOptions.Singleline | RegexOptions.Compiled);
    private static readonly Regex ThinkBlockRegex = new(@"<think>.*?</think>", RegexOptions.Singleline | RegexOptions.Compiled);
    private static readonly Regex HtmlPTagRegex = new(@"</p\s*>|<p\s*>|<p/>", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex HtmlBrTagRegex = new(@"</br\s*>|<br\s*>|<br/>", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex CjkBetweenRegex = new("(?<=[一-龥])\\s+(?=[一-龥])", RegexOptions.Compiled);
    private static readonly Regex CjkBeforeAsciiRegex = new("(?<=[一-龥])\\s+(?=[a-zA-Z0-9\\(\\)\\[\\]@#$%!&\\*\\-=+_])", RegexOptions.Compiled);
    private static readonly Regex AsciiBeforeCjkRegex = new("(?<=[a-zA-Z0-9\\(\\)\\[\\]@#$%!&\\*\\-=+_])\\s+(?=[一-龥])", RegexOptions.Compiled);
    private static readonly Regex QuoteBeforeCjkRegex = new("['\"]+(?=[一-龥])", RegexOptions.Compiled);
    private static readonly Regex QuoteAfterCjkRegex = new("(?<=[一-龥])['\"]+", RegexOptions.Compiled);
    private static readonly Regex NarrowNbspAfterNonDigitRegex = new("(?<=[^\\d]) ", RegexOptions.Compiled);
    private static readonly Regex PureDigitsRegex = new(@"^[0-9]+$", RegexOptions.Compiled);

    /// <summary>Split a string by multiple literal markers, trimming and dropping empties.</summary>
    public static List<string> SplitStringByMultiMarkers(string? content, IReadOnlyList<string> markers)
    {
        content ??= string.Empty;
        if (markers.Count == 0)
        {
            return [content];
        }

        var pattern = string.Join("|", markers.Select(Regex.Escape));
        var parts = Regex.Split(content, pattern);
        var result = new List<string>(parts.Length);
        foreach (var part in parts)
        {
            var trimmed = part.Trim();
            if (trimmed.Length > 0)
            {
                result.Add(trimmed);
            }
        }
        return result;
    }

    /// <summary>Whether the value matches a simple signed decimal number.</summary>
    public static bool IsFloatRegex(string value) => FloatRegex.IsMatch(value);

    /// <summary>
    /// Truncate a list so the cumulative token count of <paramref name="keySelector"/> stays
    /// within <paramref name="maxTokenSize"/>. Ported from <c>truncate_list_by_token_size</c>.
    /// </summary>
    public static IReadOnlyList<T> TruncateListByTokenSize<T>(
        IReadOnlyList<T> data,
        Func<T, string> keySelector,
        int maxTokenSize,
        ITokenizer tokenizer)
    {
        if (maxTokenSize <= 0)
        {
            return [];
        }

        var tokens = 0;
        for (var i = 0; i < data.Count; i++)
        {
            tokens += tokenizer.Encode(keySelector(data[i])).Count;
            if (tokens > maxTokenSize)
            {
                return data.Take(i).ToList();
            }
        }
        return data;
    }

    /// <summary>Remove &lt;think&gt;...&lt;/think&gt; blocks (and orphaned leading &lt;/think&gt;).</summary>
    public static string RemoveThinkTags(string text)
    {
        text = OrphanThinkPrefixRegex.Replace(text, "");
        text = ThinkBlockRegex.Replace(text, "");
        return text.Trim();
    }

    /// <summary>
    /// Sanitize then normalize extracted text, ported from <c>sanitize_and_normalize_extracted_text</c>:
    /// HTML-unescape, then run <see cref="NormalizeExtractedInfo"/>.
    /// </summary>
    public static string SanitizeAndNormalize(string text, bool removeInnerQuotes = false)
    {
        if (string.IsNullOrEmpty(text))
        {
            return string.Empty;
        }
        var decoded = System.Net.WebUtility.HtmlDecode(text).Trim();
        return NormalizeExtractedInfo(decoded, removeInnerQuotes);
    }

    private static readonly (string From, string To)[] FullWidthLetterMap =
    [
        ("ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        ("ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ", "abcdefghijklmnopqrstuvwxyz"),
        ("０１２３４５６７８９", "0123456789"),
    ];

    /// <summary>
    /// Normalize an entity/relation name or description, ported from <c>normalize_extracted_info</c>.
    /// Handles HTML tag stripping, full-width→half-width conversion, CJK spacing, outer-quote
    /// stripping, and numeric noise filtering.
    /// </summary>
    public static string NormalizeExtractedInfo(string name, bool removeInnerQuotes = false)
    {
        name = HtmlPTagRegex.Replace(name, "");
        name = HtmlBrTagRegex.Replace(name, "");

        // Full-width letters/numbers to half-width.
        foreach (var (from, to) in FullWidthLetterMap)
        {
            var sb = new StringBuilder(name.Length);
            foreach (var ch in name)
            {
                var idx = from.IndexOf(ch);
                sb.Append(idx >= 0 ? to[idx] : ch);
            }
            name = sb.ToString();
        }

        // Full-width symbols to half-width.
        name = name.Replace('－', '-').Replace('＋', '+').Replace('／', '/').Replace('＊', '*');
        name = name.Replace('（', '(').Replace('）', ')');
        name = name.Replace('—', '-');
        name = name.Replace('　', ' ');

        // Remove spaces between Chinese characters and around Chinese/ASCII boundaries.
        name = CjkBetweenRegex.Replace(name, "");
        name = CjkBeforeAsciiRegex.Replace(name, "");
        name = AsciiBeforeCjkRegex.Replace(name, "");

        // Strip matching outer quotes when there is no same quote inside.
        if (name.Length >= 2)
        {
            name = StripOuterQuote(name, '"', '"');
            name = StripOuterQuote(name, '\'', '\'');
            name = StripOuterQuote(name, '“', '”'); // “ ”
            name = StripOuterQuote(name, '‘', '’'); // ‘ ’
            name = StripOuterQuote(name, '《', '》'); // 《 》
        }

        if (removeInnerQuotes)
        {
            name = name.Replace("“", "").Replace("”", "").Replace("‘", "").Replace("’", "");
            name = QuoteBeforeCjkRegex.Replace(name, "");
            name = QuoteAfterCjkRegex.Replace(name, "");
            name = name.Replace(' ', ' '); // non-breaking space -> regular space
            name = NarrowNbspAfterNonDigitRegex.Replace(name, " ");
        }

        name = name.Trim();

        // Filter pure-numeric content shorter than 3 chars.
        if (name.Length < 3 && PureDigitsRegex.IsMatch(name))
        {
            return "";
        }

        // Filter short dot+digit noise (e.g. "1.2", ".123") shorter than 6 chars.
        if (name.Length < 6 && name.Length > 0 && name.All(c => char.IsDigit(c) || c == '.') && name.Contains('.'))
        {
            return "";
        }

        return name;
    }

    private static string StripOuterQuote(string name, char open, char close)
    {
        if (name.Length >= 2 && name[0] == open && name[^1] == close)
        {
            var inner = name[1..^1];
            var hasInner = open == close
                ? inner.Contains(open)
                : inner.Contains(open) || inner.Contains(close);
            if (!hasInner)
            {
                return inner;
            }
        }
        return name;
    }
}
