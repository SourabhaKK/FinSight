# Base image pinned by digest (tag: python:3.11-slim); Dependabot keeps it current.
FROM python:3.11-slim@sha256:65bd3d9f1d7fa268dd1909973ba7eb4bdf41f6dcf2cfbe76ba9528b170f3fd79 AS builder

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml ./

# Append CPU-only torch index so uv resolves the ~180 MB wheel instead of the
# default GPU wheel (+1.5 GB of CUDA packages). Modifies only the build-layer copy.
RUN printf '\n[[tool.uv.index]]\nname = "pytorch-cpu"\nurl = "https://download.pytorch.org/whl/cpu"\nexplicit = true\n\n[tool.uv.sources]\ntorch = { index = "pytorch-cpu" }\n' >> pyproject.toml

RUN uv sync --no-dev

FROM python:3.11-slim@sha256:65bd3d9f1d7fa268dd1909973ba7eb4bdf41f6dcf2cfbe76ba9528b170f3fd79

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY src/ ./src/

RUN useradd --system --no-create-home --uid 10001 app \
    && mkdir -p artefacts \
    && chown -R app:app /app

USER app

EXPOSE 8000

ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
