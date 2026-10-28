#!/usr/bin/env sh
# Run a command in the development container with the project mounted at /work.
# Example: scripts/dev.sh python -m jobnlp.cli market
docker run --rm -v "$(pwd)":/work -w /work --add-host host.docker.internal:host-gateway \
  jobnlp-dev "$@"
