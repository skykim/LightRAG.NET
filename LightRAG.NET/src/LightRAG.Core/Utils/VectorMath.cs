namespace LightRAG.Core.Utils;

/// <summary>Small vector helpers for cosine similarity, replacing the numpy ops used by nano-vectordb.</summary>
public static class VectorMath
{
    public static double Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Vector length mismatch: {a.Length} vs {b.Length}.");
        }
        double sum = 0;
        for (var i = 0; i < a.Length; i++)
        {
            sum += (double)a[i] * b[i];
        }
        return sum;
    }

    public static double Norm(ReadOnlySpan<float> v)
    {
        double sum = 0;
        for (var i = 0; i < v.Length; i++)
        {
            sum += (double)v[i] * v[i];
        }
        return Math.Sqrt(sum);
    }

    /// <summary>Cosine similarity in [-1, 1]. Pass a precomputed norm for <paramref name="a"/> to avoid recomputation.</summary>
    public static double CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b, double? normA = null)
    {
        var na = normA ?? Norm(a);
        var nb = Norm(b);
        if (na == 0 || nb == 0)
        {
            return 0;
        }
        return Dot(a, b) / (na * nb);
    }
}
