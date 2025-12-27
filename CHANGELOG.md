# Changelog

Todas as mudanças notáveis deste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Adicionado
- Sistema de rate limiting configurável via environment variables
- Hierarquia de exceções customizadas (`RAGException`, `DatabaseError`, etc.)
- Logger estruturado com suporte a JSON e cores no console
- Mensagens padronizadas em PT-BR (`core/messages.py`)
- ConnectionPool para SQLite com suporte a APSW e sqlite-vec
- Configuração centralizada via `RAGConfig.from_env()`
- mypy configurado no CI para type checking progressivo
- Pre-commit hooks com Husky + lint-staged (frontend)

### Alterado
- TTL do cache agora é configurável via `EMBEDDING_CACHE_TTL` e `RESPONSE_CACHE_TTL`
- Rate limits configuráveis via `RATE_LIMIT_CHAT`, `RATE_LIMIT_SEARCH`, etc.
- Validação de path traversal melhorada com `Path.resolve()`
- Dependências consolidadas em `pyproject.toml`

### Corrigido
- API keys não são mais logadas em plain text
- Path traversal attack prevention melhorado

## [0.1.0] - 2025-12-27

### Adicionado
- RAG SDK inicial com suporte a Claude Agent SDK
- Busca híbrida (BM25 + Vector)
- Reranking com cross-encoder
- Sistema de chunking adaptativo
- Integração com AgentFS para persistência
- Endpoints REST + SSE streaming
- Autenticação via API Key
- Sistema de sessões com histórico
- Auditoria de tool calls
- Cache LRU para embeddings e respostas
- Circuit breaker para proteção contra falhas em cascata
- Prompt Guard para detecção de prompt injection
- Suporte a MCP (Model Context Protocol)
- Docker multi-stage build
- CI/CD com GitHub Actions

### Dependências
- FastAPI >= 2.0
- Claude Agent SDK
- AgentFS SDK >= 0.4.0
- FastEmbed >= 0.3.0
- SQLite-vec >= 0.1.0
- APSW >= 3.45.0
