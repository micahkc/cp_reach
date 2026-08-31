# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# cp_reach JupyterLab image
#
# A self-contained, offline-capable JupyterLab environment with the full
# cp_reach stack pre-installed. Run the container, open the printed URL, and
# every example notebook in examples/ works with no network access.
#
# Base choice is forced by rumoca's only Linux wheel
# (rumoca-*-cp312-cp312-manylinux_2_39_x86_64.whl):
#   * Python 3.12          -> Ubuntu 24.04 ships 3.12 as the system python
#   * glibc >= 2.39        -> Ubuntu 24.04 (noble) is glibc 2.39
#   * x86_64 only          -> this image is amd64-only (no rumoca arm64 wheel)
# Debian bookworm-based python images (glibc 2.36) cannot install that wheel.
# ---------------------------------------------------------------------------
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Notebooks use the inline backend; this is just a safe default for any
    # plain scripts run in the container.
    MPLBACKEND=inline

# System packages: the Python 3.12 toolchain plus the native libraries the
# scientific stack loads at runtime (matplotlib backends, graph rendering).
# tini gives us correct PID 1 signal handling for the long-lived server.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        git \
        graphviz \
        libgl1 \
        libglib2.0-0 \
        tini \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# A dedicated virtualenv keeps us clear of Ubuntu's PEP 668 "externally
# managed" system Python and makes the whole install self-contained.
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Non-root user that owns both the workspace and the virtualenv.
ARG NB_USER=cpreach
ARG NB_UID=1001
RUN useradd --create-home --uid ${NB_UID} ${NB_USER} \
    && chown -R ${NB_USER}:${NB_USER} "$VIRTUAL_ENV"

WORKDIR /home/${NB_USER}/work

# Bring in the whole project (notebooks included) so the image is usable
# offline out of the box. .dockerignore keeps .git/.venv/etc. out.
COPY --chown=${NB_USER}:${NB_USER} . /home/${NB_USER}/work

# Pre-bake every dependency so the running container needs no network:
#   cp_reach[all] + rumoca        -> the library and its full feature set
#   jupyterlab + ipykernel        -> the server/editor itself
#   jupyterlab-lsp + pylsp        -> offline code intelligence (completion,
#                                    hover, diagnostics) for .py and notebooks
#   ipympl                        -> interactive matplotlib widgets in the lab
RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[all]" \
        rumoca \
        jupyterlab \
        ipykernel \
        ipympl \
        jupyterlab-lsp \
        "python-lsp-server[all]" \
    && python -m ipykernel install --sys-prefix \
        --name cp_reach --display-name "Python (cp_reach)"

# The editable install (run as root) and the WORKDIR-created directory leave
# root-owned paths under the home dir. Hand the whole home back to the non-root
# user so the running server can write notebooks, caches, and outputs.
RUN chown -R ${NB_USER}:${NB_USER} /home/${NB_USER}

USER ${NB_USER}
EXPOSE 8888

# tini as PID 1; JupyterLab bound to all interfaces. Auth keeps Jupyter's
# default token (printed to the logs on startup) — do not disable it.
ENTRYPOINT ["tini", "-g", "--"]
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--ServerApp.root_dir=/home/cpreach/work"]
