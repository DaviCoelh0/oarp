# Sandbox image for running task tests on model-written code (see
# src/oarp/envs/coding/sandbox.py). Build once on the host:
#   docker build -t oarp-sandbox:0.1 -f docker/sandbox.Dockerfile docker
# The base image is pinned by digest (python:3.12-slim, Python 3.12.15, pulled
# 2026-10-21) and every Python package by version and hash, so the
# environment cannot drift between runs.
FROM python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2

COPY sandbox-requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Unprivileged user (uid/gid 65534 = "nobody"); the runner also passes --user.
USER 65534:65534
WORKDIR /work
