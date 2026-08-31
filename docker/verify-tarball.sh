#!/usr/bin/env bash
#
# Verify a cp_reach JupyterLab image tarball: integrity -> load -> runs offline.
# Run this after transferring the tarball to a new machine to confirm it works
# before relying on it.
#
# Usage:
#   docker/verify-tarball.sh [path-to.tar.gz]   # default: dist/cp_reach-jupyter-0.2.1.tar.gz
#   docker/verify-tarball.sh --full [tarball]   # also run the test suite in-image (slow)
#
set -euo pipefail

FULL=0
if [[ "${1:-}" == "--full" ]]; then FULL=1; shift; fi
TARBALL="${1:-dist/cp_reach-jupyter-0.2.1.tar.gz}"

fail() { echo "FAIL: $*" >&2; exit 1; }

[[ -f "${TARBALL}" ]] || fail "tarball not found: ${TARBALL}"

echo "==> [1/5] checksum"
if [[ -f "${TARBALL}.sha256" ]]; then
  ( cd "$(dirname "${TARBALL}")" && sha256sum -c "$(basename "${TARBALL}").sha256" ) \
    || fail "checksum mismatch (file corrupt or modified)"
else
  echo "    (no ${TARBALL}.sha256 alongside — skipping checksum)"
fi

echo "==> [2/5] gzip integrity"
gzip -t "${TARBALL}" || fail "gzip reports a truncated/corrupt archive"

echo "==> [3/5] docker load"
# Capture the image reference that 'docker load' reports (e.g. cp_reach-jupyter:0.2.1).
LOAD_OUT="$(docker load --input "${TARBALL}")"
echo "    ${LOAD_OUT}"
IMAGE="$(printf '%s\n' "${LOAD_OUT}" | sed -n 's/^Loaded image: //p' | head -n1)"
[[ -n "${IMAGE}" ]] || fail "could not determine loaded image name"

echo "==> [4/5] runs with NO network (imports + JupyterLab boots)"
docker run --rm --network none "${IMAGE}" \
  python -c "import cp_reach, rumoca, cyecca, jupyterlab; print('    imports OK')" \
  || fail "offline imports failed"
docker run --rm --network none "${IMAGE}" \
  bash -lc "timeout 25 jupyter lab --ip=127.0.0.1 --no-browser 2>&1 | grep -m1 'is running at'" \
  || fail "JupyterLab did not start offline"

echo "==> [5/5] test suite in-image"
if [[ "${FULL}" -eq 1 ]]; then
  docker run --rm --network none "${IMAGE}" pytest tests/ -m "not slow" -q \
    || fail "test suite failed in-image"
else
  echo "    (skipped — pass --full to run it)"
fi

echo
echo "PASS: ${IMAGE} loaded and runs offline."
echo "Start it with:  docker run --rm -p 8888:8888 ${IMAGE}"
