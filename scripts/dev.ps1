# Run a command in the development container with the project mounted at /work.
# Example: .\scripts\dev.ps1 python -m jobnlp.cli market
$root = Split-Path -Parent $PSScriptRoot
docker run --rm -v "${root}:/work" -w /work --add-host host.docker.internal:host-gateway jobnlp-dev @args
