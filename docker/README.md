# cp_reach JupyterLab image

A self-contained, **offline-capable** JupyterLab environment with the full
`cp_reach` stack (including RuMoCA, Cyecca, and the LMI/optimization
toolchain) pre-installed. Run the container, open the printed URL, and use the
CI-gated notebooks with no network access.

## Pulling the released image

Images are built and published to the GitHub Container Registry on every
release by `.github/workflows/jupyterlab-image.yml`:

```bash
docker pull ghcr.io/<owner>/cp_reach-jupyter:latest      # or a version tag, e.g. :0.2.0
```

The canonical GHCR package is currently private. Authenticate with an account
that has package access before pulling it. Access can be granted to STR
evaluators.

## Running

```bash
docker run --rm -p 8888:8888 ghcr.io/<owner>/cp_reach-jupyter:latest
```

Then open the URL printed in the logs — it includes the one-time auth token,
e.g. `http://127.0.0.1:8888/lab?token=...`. To persist your edits to the
host, mount a working directory over the baked-in copy:

```bash
docker run --rm -p 8888:8888 -v "$PWD:/home/cpreach/work" \
  ghcr.io/<owner>/cp_reach-jupyter:latest
```

### Fully offline / air-gapped use

Nothing is fetched at runtime — the editor, all extensions, and every
dependency are baked into the image at build time. To move it to an
air-gapped host:

```bash
docker save ghcr.io/<owner>/cp_reach-jupyter:latest | gzip > cp_reach-jupyter.tar.gz
# transfer the file, then on the target host:
docker load < cp_reach-jupyter.tar.gz
```

## What's inside

- **Base:** Ubuntu 24.04 + Python 3.12 with RuMoCA 0.10. The published image is
  currently built and tested for **amd64** only.
- **Editor:** JupyterLab with `jupyterlab-lsp` + `python-lsp-server` for
  offline code intelligence (completion, hover, diagnostics), and `ipympl`
  for interactive matplotlib widgets.
- **Kernel:** `Python (cp_reach)`, the project installed editable so changes
  to a mounted source tree take effect immediately.

## Building locally

```bash
docker build -f docker/Dockerfile -t cp_reach-jupyter:dev .
```
