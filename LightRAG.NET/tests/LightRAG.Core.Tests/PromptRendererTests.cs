using LightRAG.Core.Prompts;

namespace LightRAG.Core.Tests;

public class PromptRendererTests
{
    [Fact]
    public void Replaces_named_placeholders()
    {
        var result = PromptRenderer.Render("Hello {name}, in {language}.", ("name", "Alex"), ("language", "English"));
        Assert.Equal("Hello Alex, in English.", result);
    }

    [Fact]
    public void Unescapes_doubled_braces()
    {
        // Mirrors the keywords prompt: "{{" should render to "{".
        var result = PromptRenderer.Render("First char must be {{ and last must be }}.");
        Assert.Equal("First char must be { and last must be }.", result);
    }

    [Fact]
    public void Leaves_literal_json_braces_in_inserted_examples_intact()
    {
        // The {examples} slot carries raw JSON with single braces that must survive rendering.
        var examples = """{"entities": [{"name": "Alex"}], "relationships": []}""";
        var result = PromptRenderer.Render("---Examples---\n{examples}\n---", ("examples", examples));
        Assert.Contains("""{"entities": [{"name": "Alex"}], "relationships": []}""", result);
    }

    [Fact]
    public void Renders_text_extraction_example_with_delimiter()
    {
        var rendered = PromptRenderer.Render(
            PromptTemplates.EntityExtractionExamples[0],
            ("tuple_delimiter", PromptTemplates.DefaultTupleDelimiter),
            ("completion_delimiter", PromptTemplates.DefaultCompletionDelimiter));

        Assert.Contains($"entity{PromptTemplates.DefaultTupleDelimiter}Alex{PromptTemplates.DefaultTupleDelimiter}Person", rendered);
        Assert.EndsWith(PromptTemplates.DefaultCompletionDelimiter, rendered.TrimEnd());
        Assert.DoesNotContain("{tuple_delimiter}", rendered);
    }
}
