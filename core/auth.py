# =============================================================================
# AUTH - Autenticação e Autorização
# =============================================================================
# Autenticação via API Key com suporte a escopos e expiração
# =============================================================================

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class AuthScope(str, Enum):
    """Escopos de autorização."""
    READ = "read"           # Buscar documentos
    WRITE = "write"         # Criar/atualizar documentos
    ADMIN = "admin"         # Acesso administrativo
    METRICS = "metrics"     # Acessar métricas


@dataclass
class APIKey:
    """Representação de uma API key."""
    key_id: str                          # Identificador público
    key_hash: str                        # Hash da chave (nunca armazenar plain)
    name: str                            # Nome/descrição
    scopes: list[AuthScope]              # Permissões
    owner: str                           # Dono da chave
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    rate_limit_override: Optional[int] = None  # Override do rate limit

    def is_valid(self) -> bool:
        """Verifica se a chave é válida."""
        if not self.is_active:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True

    def has_scope(self, scope: AuthScope) -> bool:
        """Verifica se tem um escopo específico."""
        return AuthScope.ADMIN in self.scopes or scope in self.scopes

    def to_dict(self) -> dict:
        """Converte para dicionário (sem hash)."""
        return {
            "key_id": self.key_id,
            "name": self.name,
            "scopes": [s.value for s in self.scopes],
            "owner": self.owner,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "is_active": self.is_active,
        }


@dataclass
class AuthResult:
    """Resultado da autenticação."""
    authenticated: bool
    api_key: Optional[APIKey] = None
    error: Optional[str] = None
    user_id: Optional[str] = None


class APIKeyManager:
    """Gerenciador de API keys."""

    # Prefixo para identificar chaves
    KEY_PREFIX = "rag_"

    def __init__(self):
        self._keys: dict[str, APIKey] = {}
        self._key_to_id: dict[str, str] = {}  # hash -> key_id

    def _hash_key(self, key: str) -> str:
        """Gera hash seguro da chave."""
        return hashlib.sha256(key.encode()).hexdigest()

    def _generate_key(self) -> tuple[str, str]:
        """
        Gera nova API key.

        Returns:
            (key_id, full_key)
        """
        key_id = secrets.token_hex(8)
        secret = secrets.token_hex(24)
        full_key = f"{self.KEY_PREFIX}{key_id}_{secret}"
        return key_id, full_key

    def create_key(
        self,
        name: str,
        owner: str,
        scopes: list[AuthScope],
        expires_in_days: Optional[int] = None,
    ) -> tuple[str, APIKey]:
        """
        Cria nova API key.

        Args:
            name: Nome descritivo
            owner: Identificador do dono
            scopes: Lista de permissões
            expires_in_days: Dias até expiração (None = nunca)

        Returns:
            (full_key, APIKey) - full_key só é retornado uma vez!
        """
        key_id, full_key = self._generate_key()
        key_hash = self._hash_key(full_key)

        expires_at = None
        if expires_in_days:
            from datetime import timedelta
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            scopes=scopes,
            owner=owner,
            expires_at=expires_at,
        )

        self._keys[key_id] = api_key
        self._key_to_id[key_hash] = key_id

        return full_key, api_key

    def authenticate(self, key: str) -> AuthResult:
        """
        Autentica usando API key.

        Args:
            key: API key completa

        Returns:
            AuthResult com status
        """
        if not key:
            return AuthResult(authenticated=False, error="No API key provided")

        # Verificar formato
        if not key.startswith(self.KEY_PREFIX):
            return AuthResult(authenticated=False, error="Invalid key format")

        # Buscar por hash
        key_hash = self._hash_key(key)
        key_id = self._key_to_id.get(key_hash)

        if not key_id:
            return AuthResult(authenticated=False, error="Invalid API key")

        api_key = self._keys.get(key_id)
        if not api_key:
            return AuthResult(authenticated=False, error="API key not found")

        # Verificar validade
        if not api_key.is_valid():
            if not api_key.is_active:
                return AuthResult(authenticated=False, error="API key is disabled")
            return AuthResult(authenticated=False, error="API key expired")

        # Atualizar último uso
        api_key.last_used_at = datetime.now(timezone.utc)

        return AuthResult(
            authenticated=True,
            api_key=api_key,
            user_id=api_key.owner,
        )

    def revoke_key(self, key_id: str) -> bool:
        """Revoga uma API key."""
        if key_id in self._keys:
            self._keys[key_id].is_active = False
            return True
        return False

    def get_key(self, key_id: str) -> Optional[APIKey]:
        """Retorna API key por ID."""
        return self._keys.get(key_id)

    def list_keys(self, owner: Optional[str] = None) -> list[APIKey]:
        """Lista API keys, opcionalmente filtradas por owner."""
        keys = list(self._keys.values())
        if owner:
            keys = [k for k in keys if k.owner == owner]
        return keys


def extract_api_key(auth_header: Optional[str]) -> Optional[str]:
    """
    Extrai API key do header Authorization.

    Formatos suportados:
    - Bearer <key>
    - ApiKey <key>
    - <key> (direto)
    """
    if not auth_header:
        return None

    auth_header = auth_header.strip()

    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    elif auth_header.startswith("ApiKey "):
        return auth_header[7:]
    elif auth_header.startswith(APIKeyManager.KEY_PREFIX):
        return auth_header

    return None


# Instância global
_key_manager: Optional[APIKeyManager] = None


def get_key_manager() -> APIKeyManager:
    """Retorna gerenciador global de API keys."""
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager()
    return _key_manager


def authenticate(auth_header: Optional[str]) -> AuthResult:
    """Autentica usando header Authorization."""
    key = extract_api_key(auth_header)
    return get_key_manager().authenticate(key) if key else AuthResult(
        authenticated=False,
        error="No authorization header"
    )


# =============================================================================
# FUNÇÕES PARA FASTAPI - Dependency Injection
# =============================================================================

import os
from pathlib import Path

# Carregar variaveis de ambiente do .env
try:
    from dotenv import load_dotenv, set_key
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(_env_path)
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False
    _env_path = None

# Chaves válidas
VALID_API_KEYS: set[str] = set()
_AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"

# Carregar API Key do .env ou gerar nova
_api_key = os.getenv("RAG_API_KEY")

if _api_key:
    # Key encontrada no .env
    VALID_API_KEYS.add(_api_key)
    print(f"[AUTH] API Key carregada do .env: {_api_key[:20]}...")
else:
    # Gerar nova key
    _api_key = f"rag_{secrets.token_urlsafe(32)}"
    VALID_API_KEYS.add(_api_key)

    # Tentar salvar no .env para persistir
    if _DOTENV_AVAILABLE and _env_path:
        try:
            _env_path.touch(exist_ok=True)
            set_key(str(_env_path), "RAG_API_KEY", _api_key)
            print(f"[AUTH] Nova API Key gerada e salva em .env")
            print(f"[AUTH] RAG_API_KEY={_api_key}")
        except Exception as e:
            print(f"[AUTH] Aviso: Nao foi possivel salvar no .env: {e}")
            print(f"[AUTH] API Key temporaria: {_api_key}")
    else:
        print(f"[AUTH] API Key temporaria (instale python-dotenv para persistir): {_api_key}")


def is_auth_enabled() -> bool:
    """Verifica se autenticação está habilitada."""
    return _AUTH_ENABLED


from fastapi import Header

async def verify_api_key(
    x_api_key: str = Header(None, alias="X-API-Key"),
    authorization: str = Header(None),
) -> str:
    """
    Dependency do FastAPI para verificar API key.

    Aceita chave via:
    - Header X-API-Key
    - Header Authorization: Bearer <key>

    Returns:
        API key se válida

    Raises:
        HTTPException 401 se inválida
    """
    from fastapi import HTTPException, Header

    # Se auth está desabilitada, retorna placeholder
    if not is_auth_enabled():
        return "auth_disabled"

    # Extrair chave do header apropriado
    key = None
    if x_api_key:
        key = x_api_key
    elif authorization:
        if authorization.startswith("Bearer "):
            key = authorization[7:]
        else:
            key = authorization

    if not key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Use X-API-Key header or Authorization: Bearer <key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verificar se é uma chave válida (simples)
    if key in VALID_API_KEYS:
        return key

    # Verificar usando o manager (para chaves criadas programaticamente)
    manager = get_key_manager()
    result = manager.authenticate(key)
    if result.authenticated:
        return key

    raise HTTPException(
        status_code=401,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def add_valid_key(key: str) -> None:
    """Adiciona uma chave válida."""
    VALID_API_KEYS.add(key)


def remove_valid_key(key: str) -> None:
    """Remove uma chave válida."""
    VALID_API_KEYS.discard(key)


if __name__ == "__main__":
    print("=== Teste de API Key Auth ===\n")

    manager = APIKeyManager()

    # Criar chaves de teste
    print("--- Criando API Keys ---")

    full_key, api_key = manager.create_key(
        name="Development Key",
        owner="dev@empresa.com",
        scopes=[AuthScope.READ, AuthScope.WRITE],
        expires_in_days=30,
    )
    print(f"Key criada: {full_key[:30]}...")
    print(f"Key ID: {api_key.key_id}")
    print(f"Scopes: {[s.value for s in api_key.scopes]}")

    admin_key, admin_api_key = manager.create_key(
        name="Admin Key",
        owner="admin@empresa.com",
        scopes=[AuthScope.ADMIN],
    )
    print(f"\nAdmin key criada: {admin_key[:30]}...")

    # Testar autenticação
    print("\n--- Testando Autenticação ---")

    # Chave válida
    result = manager.authenticate(full_key)
    print(f"Chave válida: {'✓' if result.authenticated else '✗'}")

    # Chave inválida
    result = manager.authenticate("rag_invalid_key")
    print(f"Chave inválida: {'✓' if not result.authenticated else '✗'} ({result.error})")

    # Verificar escopos
    print("\n--- Verificando Escopos ---")
    print(f"Dev key has READ: {api_key.has_scope(AuthScope.READ)}")
    print(f"Dev key has ADMIN: {api_key.has_scope(AuthScope.ADMIN)}")
    print(f"Admin key has READ: {admin_api_key.has_scope(AuthScope.READ)}")  # True (admin tem tudo)

    # Testar revogação
    print("\n--- Testando Revogação ---")
    manager.revoke_key(api_key.key_id)
    result = manager.authenticate(full_key)
    print(f"Após revogação: {'✓' if not result.authenticated else '✗'} ({result.error})")
