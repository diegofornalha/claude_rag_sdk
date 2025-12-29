# =============================================================================
# DOCKERFILE - Claude RAG SDK Backend
# =============================================================================
# Multi-stage build otimizado para produção
# Imagem final: ~300MB (vs ~1.2GB sem multi-stage)
# =============================================================================

# -----------------------------------------------------------------------------
# STAGE 1: Builder - Instala dependências e compila
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /app

# Instalar dependências de build (necessárias para apsw, sqlite-vec)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Criar virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar dependências Python primeiro (cache layer)
COPY pyproject.toml ./
COPY README.md ./

# Instalar dependências (sem o projeto em si ainda)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Copiar código fonte
COPY claude_rag_sdk/ claude_rag_sdk/
COPY routers/ routers/
COPY server.py app_state.py ./

# Reinstalar com código fonte
RUN pip install --no-cache-dir -e .

# -----------------------------------------------------------------------------
# STAGE 2: Runtime - Imagem final mínima
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

WORKDIR /app

# Labels para metadados
LABEL maintainer="Claude Partner"
LABEL version="0.1.0"
LABEL description="Claude RAG SDK - Semantic search and AI-powered Q&A"

# Criar usuário não-root para segurança
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copiar virtual environment do builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copiar código fonte
COPY --from=builder /app/claude_rag_sdk ./claude_rag_sdk
COPY --from=builder /app/routers ./routers
COPY --from=builder /app/server.py ./
COPY --from=builder /app/app_state.py ./
COPY --from=builder /app/pyproject.toml ./

# Criar diretórios necessários com permissões corretas
RUN mkdir -p /app/data /app/artifacts /app/.agentfs && \
    chown -R appuser:appgroup /app

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8001

# Trocar para usuário não-root
USER appuser

# Expor porta
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# Comando de inicialização
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
