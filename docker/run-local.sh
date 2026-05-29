#!/usr/bin/env bash
#
# Build and run the cp_reach JupyterLab image locally.
#
# Usage:
#   docker/run-local.sh              # build + run, JupyterLab on http://localhost:8888
#   PORT=9000 docker/run-local.sh    # use a different host port
#   docker/run-local.sh --offline    # build, then verify it runs with no network
#
set -euo pipefail

# Resolve the repo root from this script's location so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="cp_reach-jupyter:dev"
PORT="${PORT:-8888}"

echo ">> Building ${IMAGE} ..."
docker build -f "${REPO_ROOT}/docker/Dockerfile" -t "${IMAGE}" "${REPO_ROOT}"

# --offline: prove the image is self-contained (imports + tests with no network).
if [[ "${1:-}" == "--offline" ]]; then
  echo ">> Verifying offline imports (--network none) ..."
  docker run --rm --network none "${IMAGE}" \
    python -c "import cp_reach, rumoca, cyecca, jupyterlab; print('offline imports OK')"
  echo ">> Running test suite offline (--network none) ..."
  docker run --rm --network none "${IMAGE}" pytest tests/ -m "not slow" -q
  echo ">> Offline checks passed."
  exit 0
fi

echo ">> Starting JupyterLab on host port ${PORT} (Ctrl-C to stop) ..."
echo ">> Open the http://127.0.0.1:8888/lab?token=... URL printed below,"
echo ">> substituting port ${PORT} if you changed it."
echo

# Foreground + --rm so the token URL is visible and the container is cleaned up
# on exit. Runs the self-contained baked copy of the repo (writable by the
# container user), so edits made in the browser do NOT persist to the host.
# To persist edits instead, run the container as your own user and bind-mount:
#   docker run --rm -it -p 8888:8888 --user "$(id -u):$(id -g)" \
#     -e HOME=/tmp -v "${REPO_ROOT}:/home/cpreach/work" "${IMAGE}"
exec docker run --rm -it \
  -p "${PORT}:8888" \
  "${IMAGE}"
