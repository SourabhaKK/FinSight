FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml ./

# Append CPU-only torch index so uv resolves the ~180 MB wheel instead of the
# default GPU wheel (+1.5 GB of CUDA packages). Modifies only the build-layer copy.
RUN printf '\n[[tool.uv.index]]\nname = "pytorch-cpu"\nurl = "https://download.pytorch.org/whl/cpu"\npriority = "explicit"\n\n[tool.uv.sources]\ntorch = { index = "pytorch-cpu" }\n' >> pyproject.toml

RUN uv sync --no-dev

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY src/ ./src/

RUN mkdir -p artefacts

EXPOSE 8000

ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
