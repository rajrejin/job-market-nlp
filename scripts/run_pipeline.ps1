# Run the whole pipeline in the development container, one step at a time.
# Build the image once:  docker build -f docker/Dockerfile.dev -t jobnlp-dev .
# Needs Ollama on the host for the last step (LLM guard evaluation).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$steps = @(
    "harvest", "crawl", "esco", "build-corpus", "train-ner", "evaluate",
    "market", "train-classifier", "evaluate-gap", "report"
)
foreach ($s in $steps) {
    Write-Host "== jobnlp $s"
    docker run --rm -v "${root}:/work" -w /work --add-host host.docker.internal:host-gateway `
        jobnlp-dev python -m jobnlp.cli $s
    if ($LASTEXITCODE -ne 0) { throw "step $s failed" }
}
