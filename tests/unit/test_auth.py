# =============================================================================
# TESTES - Auth Module
# =============================================================================
# Testes unitários para autenticação e autorização
# =============================================================================

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta


class TestAPIKeyManager:
    """Testes para APIKeyManager."""

    def test_create_key_returns_tuple(self):
        """Verifica se create_key retorna (full_key, APIKey)."""
        from claude_rag_sdk.core.auth import APIKeyManager, AuthScope

        manager = APIKeyManager()
        full_key, api_key = manager.create_key(
            name="Test Key",
            owner="test@example.com",
            scopes=[AuthScope.READ],
        )

        assert full_key.startswith("rag_")
        assert api_key.name == "Test Key"
        assert api_key.owner == "test@example.com"
        assert AuthScope.READ in api_key.scopes

    def test_create_key_with_expiration(self):
        """Verifica criação de key com expiração."""
        from claude_rag_sdk.core.auth import APIKeyManager, AuthScope

        manager = APIKeyManager()
        _, api_key = manager.create_key(
            name="Expiring Key",
            owner="test@example.com",
            scopes=[AuthScope.READ],
            expires_in_days=30,
        )

        assert api_key.expires_at is not None
        assert api_key.expires_at > datetime.now(timezone.utc)

    def test_authenticate_valid_key(self):
        """Verifica autenticação com chave válida."""
        from claude_rag_sdk.core.auth import APIKeyManager, AuthScope

        manager = APIKeyManager()
        full_key, _ = manager.create_key(
            name="Auth Test",
            owner="test@example.com",
            scopes=[AuthScope.READ],
        )

        result = manager.authenticate(full_key)

        assert result.authenticated is True
        assert result.api_key is not None
        assert result.user_id == "test@example.com"

    def test_authenticate_invalid_key(self):
        """Verifica rejeição de chave inválida."""
        from claude_rag_sdk.core.auth import APIKeyManager

        manager = APIKeyManager()
        result = manager.authenticate("rag_invalid_key_12345")

        assert result.authenticated is False
        assert "Invalid" in result.error

    def test_authenticate_wrong_prefix(self):
        """Verifica rejeição de chave com prefixo errado."""
        from claude_rag_sdk.core.auth import APIKeyManager

        manager = APIKeyManager()
        result = manager.authenticate("wrong_prefix_key")

        assert result.authenticated is False
        assert "format" in result.error.lower()

    def test_authenticate_empty_key(self):
        """Verifica rejeição de chave vazia."""
        from claude_rag_sdk.core.auth import APIKeyManager

        manager = APIKeyManager()
        result = manager.authenticate("")

        assert result.authenticated is False

    def test_revoke_key(self):
        """Verifica revogação de chave."""
        from claude_rag_sdk.core.auth import APIKeyManager, AuthScope

        manager = APIKeyManager()
        full_key, api_key = manager.create_key(
            name="Revoke Test",
            owner="test@example.com",
            scopes=[AuthScope.READ],
        )

        # Revogar
        success = manager.revoke_key(api_key.key_id)
        assert success is True

        # Tentar autenticar
        result = manager.authenticate(full_key)
        assert result.authenticated is False
        assert "disabled" in result.error.lower()

    def test_list_keys(self):
        """Verifica listagem de chaves."""
        from claude_rag_sdk.core.auth import APIKeyManager, AuthScope

        manager = APIKeyManager()

        # Criar múltiplas chaves
        manager.create_key("Key 1", "user1@example.com", [AuthScope.READ])
        manager.create_key("Key 2", "user1@example.com", [AuthScope.WRITE])
        manager.create_key("Key 3", "user2@example.com", [AuthScope.ADMIN])

        # Listar todas
        all_keys = manager.list_keys()
        assert len(all_keys) == 3

        # Listar por owner
        user1_keys = manager.list_keys(owner="user1@example.com")
        assert len(user1_keys) == 2


class TestAPIKey:
    """Testes para dataclass APIKey."""

    def test_is_valid_active_key(self):
        """Verifica key ativa é válida."""
        from claude_rag_sdk.core.auth import APIKey, AuthScope

        key = APIKey(
            key_id="test123",
            key_hash="hash123",
            name="Test",
            scopes=[AuthScope.READ],
            owner="test@example.com",
            is_active=True,
        )

        assert key.is_valid() is True

    def test_is_valid_inactive_key(self):
        """Verifica key inativa é inválida."""
        from claude_rag_sdk.core.auth import APIKey, AuthScope

        key = APIKey(
            key_id="test123",
            key_hash="hash123",
            name="Test",
            scopes=[AuthScope.READ],
            owner="test@example.com",
            is_active=False,
        )

        assert key.is_valid() is False

    def test_is_valid_expired_key(self):
        """Verifica key expirada é inválida."""
        from claude_rag_sdk.core.auth import APIKey, AuthScope

        key = APIKey(
            key_id="test123",
            key_hash="hash123",
            name="Test",
            scopes=[AuthScope.READ],
            owner="test@example.com",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )

        assert key.is_valid() is False

    def test_has_scope_direct(self):
        """Verifica escopo direto."""
        from claude_rag_sdk.core.auth import APIKey, AuthScope

        key = APIKey(
            key_id="test123",
            key_hash="hash123",
            name="Test",
            scopes=[AuthScope.READ, AuthScope.WRITE],
            owner="test@example.com",
        )

        assert key.has_scope(AuthScope.READ) is True
        assert key.has_scope(AuthScope.WRITE) is True
        assert key.has_scope(AuthScope.ADMIN) is False

    def test_has_scope_admin_has_all(self):
        """Verifica admin tem todos os escopos."""
        from claude_rag_sdk.core.auth import APIKey, AuthScope

        key = APIKey(
            key_id="test123",
            key_hash="hash123",
            name="Admin Key",
            scopes=[AuthScope.ADMIN],
            owner="admin@example.com",
        )

        assert key.has_scope(AuthScope.READ) is True
        assert key.has_scope(AuthScope.WRITE) is True
        assert key.has_scope(AuthScope.METRICS) is True

    def test_to_dict_no_hash(self):
        """Verifica to_dict não expõe hash."""
        from claude_rag_sdk.core.auth import APIKey, AuthScope

        key = APIKey(
            key_id="test123",
            key_hash="secret_hash_value",
            name="Test",
            scopes=[AuthScope.READ],
            owner="test@example.com",
        )

        data = key.to_dict()

        assert "key_hash" not in data
        assert data["key_id"] == "test123"
        assert data["name"] == "Test"


class TestExtractAPIKey:
    """Testes para extract_api_key."""

    def test_bearer_format(self):
        """Verifica extração de Bearer token."""
        from claude_rag_sdk.core.auth import extract_api_key

        key = extract_api_key("Bearer rag_mykey123")
        assert key == "rag_mykey123"

    def test_apikey_format(self):
        """Verifica extração de ApiKey format."""
        from claude_rag_sdk.core.auth import extract_api_key

        key = extract_api_key("ApiKey rag_mykey123")
        assert key == "rag_mykey123"

    def test_direct_key(self):
        """Verifica extração de chave direta."""
        from claude_rag_sdk.core.auth import extract_api_key

        key = extract_api_key("rag_mykey123")
        assert key == "rag_mykey123"

    def test_empty_header(self):
        """Verifica header vazio."""
        from claude_rag_sdk.core.auth import extract_api_key

        key = extract_api_key("")
        assert key is None

    def test_none_header(self):
        """Verifica header None."""
        from claude_rag_sdk.core.auth import extract_api_key

        key = extract_api_key(None)
        assert key is None

    def test_invalid_format(self):
        """Verifica formato inválido."""
        from claude_rag_sdk.core.auth import extract_api_key

        key = extract_api_key("InvalidFormat token")
        assert key is None
