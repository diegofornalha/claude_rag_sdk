# Guia de Contribuição

Obrigado pelo interesse em contribuir com o Claude RAG SDK!

## Configuração do Ambiente

### Pré-requisitos

- Python >= 3.10
- pip ou uv (recomendado)

### Instalação

```bash
# Clone o repositório
git clone https://github.com/your-org/claude-rag-sdk.git
cd claude-rag-sdk/backend

# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate  # Windows

# Instalar dependências de desenvolvimento
pip install -e ".[dev]"
```

## Padrões de Código

### Formatação e Linting

Usamos **Ruff** para formatação e linting:

```bash
# Verificar código
ruff check .

# Formatar código
ruff format .

# Corrigir automaticamente
ruff check --fix .
```

### Type Hints

Usamos **mypy** para verificação de tipos:

```bash
# Verificar tipos
mypy claude_rag_sdk/ routers/ utils/ --ignore-missing-imports
```

### Testes

Usamos **pytest** para testes:

```bash
# Rodar todos os testes
pytest

# Com cobertura
pytest --cov=claude_rag_sdk --cov-report=term-missing

# Testes específicos
pytest tests/unit/test_search.py -v
pytest -k "test_chat" -v
```

## Estrutura do Projeto

```
backend/
├── claude_rag_sdk/          # SDK principal
│   ├── core/                # Infraestrutura (auth, cache, logger, etc.)
│   ├── mcp/                 # Model Context Protocol
│   └── mcp_adapters/        # Adaptadores MCP
├── routers/                 # Endpoints FastAPI
├── utils/                   # Utilitários
├── tests/                   # Testes
│   ├── unit/               # Testes unitários
│   ├── integration/        # Testes de integração
│   ├── e2e/                # Testes end-to-end
│   └── performance/        # Testes de performance
└── scripts/                 # Scripts utilitários
```

## Convenções

### Commits

Seguimos [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: adicionar busca híbrida
fix: corrigir vazamento de memória no cache
docs: atualizar README com exemplos
test: adicionar testes para reranker
refactor: extrair lógica de chunking
```

### Branches

- `main` - Produção estável
- `develop` - Desenvolvimento
- `feature/*` - Novas features
- `fix/*` - Correções de bugs
- `docs/*` - Documentação

### Pull Requests

1. Crie uma branch a partir de `develop`
2. Faça suas alterações
3. Escreva/atualize testes
4. Verifique que `ruff check .` passa
5. Verifique que `pytest` passa
6. Abra um PR para `develop`

## Adicionando Novos Endpoints

1. Crie o router em `routers/`
2. Adicione rate limiting com `@limiter.limit()`
3. Use exceções customizadas de `core/exceptions.py`
4. Adicione testes em `tests/`
5. Documente o endpoint no OpenAPI (docstring)

Exemplo:

```python
from fastapi import APIRouter, Request
from claude_rag_sdk.core.rate_limiter import RATE_LIMITS, limiter
from claude_rag_sdk.core.exceptions import ValidationError

router = APIRouter(prefix="/my-feature", tags=["My Feature"])

@router.post("/action")
@limiter.limit(RATE_LIMITS.get("default", "60/minute"))
async def my_action(request: Request, param: str):
    """Descrição do endpoint para OpenAPI."""
    if not param:
        raise ValidationError(message="Parâmetro obrigatório")
    return {"result": "ok"}
```

## Adicionando Exceções

Estenda a hierarquia em `core/exceptions.py`:

```python
class MyCustomError(RAGException):
    """Meu erro customizado."""
    http_status: int = 400
    default_message: str = "Algo deu errado"
```

## Perguntas?

Abra uma issue com a tag `question`.
