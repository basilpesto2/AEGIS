FROM python:3.11.15-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-runtime.constraints.txt /app/

RUN python -m pip install --no-cache-dir \
      --constraint /app/requirements-runtime.constraints.txt \
      torch==2.11.0 torchvision==0.26.0 \
      --index-url ${TORCH_INDEX_URL}

COPY pyproject.toml README.md /app/
COPY AEGIS /app/AEGIS
RUN python -m pip install --no-cache-dir \
      --constraint /app/requirements-runtime.constraints.txt \
      setuptools==79.0.1 wheel==0.46.3
RUN python -m pip install --no-cache-dir \
      --no-build-isolation \
      --constraint /app/requirements-runtime.constraints.txt \
      ".[mllm]"

COPY configs /app/configs
COPY models/aegis /app/models/aegis

# Keep packaged code, configurations, and detector artifacts root-owned so a
# restarted worker cannot consume a runtime-modified file after preflight.
RUN useradd --create-home --uid 10001 aegis \
    && mkdir -p /app/models/huggingface \
    && mkdir -p /var/lib/aegis/audit \
    && chown -R aegis:aegis /app/models/huggingface /var/lib/aegis/audit

USER aegis

EXPOSE 8766

HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
  CMD python -c "import json,urllib.request; o=urllib.request.build_opener(urllib.request.ProxyHandler({})); assert json.load(o.open('http://127.0.0.1:8766/readyz', timeout=3))['ok']"

ENTRYPOINT ["aegis"]
CMD ["serve", "--config", "/app/configs/aegis.deployment.container.json"]
