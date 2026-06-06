#!/usr/bin/env bash
# Build the LightRAG.NET engine (netstandard2.1) and copy the engine + dependency DLLs into the
# lightrag-unity project's Assets/Plugins/LightRAG folder. Newtonsoft.Json is intentionally excluded
# because Unity provides it via the com.unity.nuget.newtonsoft-json package (adding the DLL here would
# create a duplicate-assembly conflict).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PLUGINS="$ROOT/../lightrag-unity/Assets/Plugins/LightRAG"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1

echo "Building netstandard2.1 engine (Release, with transitive deps)..."
dotnet build "$ROOT/src/LightRAG.Storage.FileBased" -c Release -p:CopyLocalLockFileAssemblies=true >/dev/null
dotnet build "$ROOT/src/LightRAG.Providers.Ollama" -c Release -p:CopyLocalLockFileAssemblies=true >/dev/null

mkdir -p "$PLUGINS"
SRC1="$ROOT/src/LightRAG.Storage.FileBased/bin/Release/netstandard2.1"
SRC2="$ROOT/src/LightRAG.Providers.Ollama/bin/Release/netstandard2.1"

echo "Copying DLLs to $PLUGINS ..."
copied=0
for f in "$SRC1"/*.dll "$SRC2"/*.dll; do
  base="$(basename "$f")"
  [ "$base" = "Newtonsoft.Json.dll" ] && continue   # provided by the Unity package
  cp -f "$f" "$PLUGINS/$base"
  copied=$((copied+1))
done

echo "Done. $(ls "$PLUGINS"/*.dll | wc -l | tr -d ' ') DLLs in plugins folder."
