# =============================================================================
# DOCUMENT MODEL - Schema com Metadados para RBAC
# =============================================================================
# Define estrutura de documentos com campos de governança e controle de acesso
# =============================================================================

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Classification(str, Enum):
    """Níveis de classificação de documentos."""
    PUBLIC = "public"           # Acesso livre
    INTERNAL = "internal"       # Apenas funcionários
    CONFIDENTIAL = "confidential"  # Áreas específicas
    RESTRICTED = "restricted"   # Somente autorizados


@dataclass
class DocumentMetadata:
    """Metadados de governança do documento."""

    # Ownership
    owner_area: str                          # Ex: "RH", "TI", "Legal", "Compliance"
    owner_email: Optional[str] = None        # Email do responsável

    # Classification
    classification: Classification = Classification.INTERNAL

    # RBAC tags para controle de acesso
    rbac_tags: list[str] = field(default_factory=list)  # Ex: ["rh:read", "ti:admin"]

    # Audit
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_by: Optional[str] = None
    updated_at: Optional[datetime] = None

    # Retention
    retention_days: Optional[int] = None     # Dias para reter (compliance)
    expires_at: Optional[datetime] = None    # Data de expiração

    # Source tracking
    source_system: Optional[str] = None      # Ex: "confluence", "sharepoint", "manual"
    source_url: Optional[str] = None         # URL original do documento

    def to_dict(self) -> dict:
        """Converte para dicionário (para JSON/DB)."""
        return {
            "owner_area": self.owner_area,
            "owner_email": self.owner_email,
            "classification": self.classification.value,
            "rbac_tags": self.rbac_tags,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "retention_days": self.retention_days,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "source_system": self.source_system,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentMetadata":
        """Cria instância a partir de dicionário."""
        return cls(
            owner_area=data.get("owner_area", "unknown"),
            owner_email=data.get("owner_email"),
            classification=Classification(data.get("classification", "internal")),
            rbac_tags=data.get("rbac_tags", []),
            created_by=data.get("created_by"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_by=data.get("updated_by"),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            retention_days=data.get("retention_days"),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            source_system=data.get("source_system"),
            source_url=data.get("source_url"),
        )


@dataclass
class Document:
    """Documento completo com conteúdo e metadados."""

    # Identificação
    id: int
    nome: str
    tipo: str                               # Ex: "pdf", "md", "txt"

    # Conteúdo
    conteudo: str
    caminho: Optional[str] = None

    # Metadados de governança
    metadata: Optional[DocumentMetadata] = None

    # Embedding info
    has_embedding: bool = False
    embedding_model: Optional[str] = None

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "id": self.id,
            "nome": self.nome,
            "tipo": self.tipo,
            "conteudo": self.conteudo,
            "caminho": self.caminho,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "has_embedding": self.has_embedding,
            "embedding_model": self.embedding_model,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Cria instância a partir de dicionário."""
        metadata = None
        if data.get("metadata"):
            metadata = DocumentMetadata.from_dict(data["metadata"])

        return cls(
            id=data["id"],
            nome=data["nome"],
            tipo=data["tipo"],
            conteudo=data["conteudo"],
            caminho=data.get("caminho"),
            metadata=metadata,
            has_embedding=data.get("has_embedding", False),
            embedding_model=data.get("embedding_model"),
        )

    @classmethod
    def from_db_row(cls, row: tuple, metadata_json: Optional[str] = None) -> "Document":
        """Cria instância a partir de row do SQLite."""
        import json

        metadata = None
        if metadata_json:
            metadata = DocumentMetadata.from_dict(json.loads(metadata_json))

        return cls(
            id=row[0],
            nome=row[1],
            tipo=row[2],
            conteudo=row[3],
            caminho=row[4] if len(row) > 4 else None,
            metadata=metadata,
        )

    def can_access(self, user_tags: list[str]) -> bool:
        """Verifica se usuário tem acesso baseado em tags RBAC."""
        if not self.metadata:
            return True  # Sem metadata = público

        if self.metadata.classification == Classification.PUBLIC:
            return True

        if not self.metadata.rbac_tags:
            return True  # Sem tags = acesso livre dentro da classificação

        # Verificar se alguma tag do usuário permite acesso
        for user_tag in user_tags:
            if user_tag in self.metadata.rbac_tags:
                return True

            # Verificar wildcard (ex: "ti:*" permite "ti:read", "ti:write")
            if ":" in user_tag and user_tag.endswith(":*"):
                prefix = user_tag[:-1]  # "ti:"
                for doc_tag in self.metadata.rbac_tags:
                    if doc_tag.startswith(prefix):
                        return True

        return False


# SQL para adicionar coluna de metadados na tabela existente
MIGRATION_SQL = """
-- Adicionar coluna de metadados JSON à tabela documentos
ALTER TABLE documentos ADD COLUMN metadata_json TEXT;

-- Índice para busca por owner_area
CREATE INDEX IF NOT EXISTS idx_documentos_owner_area
ON documentos(json_extract(metadata_json, '$.owner_area'));

-- Índice para busca por classification
CREATE INDEX IF NOT EXISTS idx_documentos_classification
ON documentos(json_extract(metadata_json, '$.classification'));
"""


if __name__ == "__main__":
    # Teste do modelo
    metadata = DocumentMetadata(
        owner_area="TI",
        owner_email="ti@empresa.com",
        classification=Classification.CONFIDENTIAL,
        rbac_tags=["ti:read", "compliance:read"],
        source_system="confluence",
    )

    doc = Document(
        id=1,
        nome="Politica de IA.md",
        tipo="md",
        conteudo="Conteúdo do documento...",
        metadata=metadata,
    )

    print("=== Documento ===")
    print(f"Nome: {doc.nome}")
    print(f"Owner: {doc.metadata.owner_area}")
    print(f"Classification: {doc.metadata.classification.value}")
    print(f"RBAC Tags: {doc.metadata.rbac_tags}")

    print("\n=== Testes de Acesso ===")
    print(f"User ['ti:read']: {doc.can_access(['ti:read'])}")  # True
    print(f"User ['rh:read']: {doc.can_access(['rh:read'])}")  # False
    print(f"User ['ti:*']: {doc.can_access(['ti:*'])}")        # True (wildcard)
    print(f"User ['admin']: {doc.can_access(['admin'])}")      # False

    print("\n=== JSON ===")
    import json
    print(json.dumps(doc.to_dict(), indent=2, default=str))
