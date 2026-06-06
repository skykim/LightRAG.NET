using System.Text;

namespace LightRAG.Storage.FileBased;

/// <summary>
/// Shared base for file-backed storage: resolves the working directory / workspace and
/// provides atomic file writes (write-temp-then-move), mirroring the commit protocol of the
/// Python JSON / NanoVectorDB / NetworkX backends without their multi-process machinery.
/// </summary>
public abstract class FileStorageBase
{
    protected FileStorageBase(string workingDir, string @namespace, string workspace)
    {
        Namespace = @namespace;
        Workspace = workspace;
        Directory = string.IsNullOrEmpty(workspace) ? workingDir : Path.Combine(workingDir, workspace);
    }

    public string Namespace { get; }

    public string Workspace { get; }

    /// <summary>Resolved directory where this storage's file(s) live.</summary>
    protected string Directory { get; }

    protected void EnsureDirectory() => System.IO.Directory.CreateDirectory(Directory);

    /// <summary>Atomically write text to <paramref name="path"/> via a temp file + move.</summary>
    protected static void AtomicWrite(string path, string content)
    {
        var tmp = path + ".tmp";
        File.WriteAllText(tmp, content, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        // netstandard2.1 has no File.Move(overwrite) overload, so replace the target explicitly.
        if (File.Exists(path))
        {
            File.Delete(path);
        }
        File.Move(tmp, path);
    }

    protected static void SweepOrphanTmp(string path)
    {
        var tmp = path + ".tmp";
        if (File.Exists(tmp))
        {
            try { File.Delete(tmp); } catch { /* best-effort cleanup */ }
        }
    }
}
