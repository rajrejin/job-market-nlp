# Take one snapshot of all sources in the development container, then rebuild the corpus.
# Safe to rerun: pages are cached per snapshot date. Docker Desktop must be running.
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root "logs\harvest_scheduled.log"
foreach ($s in @("harvest", "crawl", "build-corpus")) {
    docker run --rm -v "${root}:/work" -w /work jobnlp-dev python -m jobnlp.cli $s *>> $log
}
