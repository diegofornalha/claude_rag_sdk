# =============================================================================
# RBAC - Role-Based Access Control para RAG Agent
# =============================================================================
# Controle de acesso baseado em roles e tags para filtragem de documentos
# =============================================================================

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json

from .logger import logger


class Role(str, Enum):
    """Roles padrão do sistema."""
    ADMIN = "admin"           # Acesso total
    ANALYST = "analyst"       # Leitura geral
    VIEWER = "viewer"         # Apenas documentos públicos
    AUDITOR = "auditor"       # Leitura + logs


@dataclass
class User:
    """Usuário com roles e tags de acesso."""

    id: str                              # user_id ou email
    roles: list[Role] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)  # Ex: ["ti:read", "rh:admin"]
    areas: list[str] = field(default_factory=list)  # Ex: ["TI", "RH"]

    def has_role(self, role: Role) -> bool:
        """Verifica se usuário tem uma role específica."""
        return role in self.roles or Role.ADMIN in self.roles

    def has_tag(self, tag: str) -> bool:
        """Verifica se usuário tem uma tag específica."""
        if Role.ADMIN in self.roles:
            return True

        if tag in self.tags:
            return True

        # Wildcard matching
        for user_tag in self.tags:
            if user_tag.endswith(":*"):
                prefix = user_tag[:-1]
                if tag.startswith(prefix):
                    return True

        return False

    def has_area_access(self, area: str) -> bool:
        """Verifica se usuário tem acesso a uma área específica."""
        if Role.ADMIN in self.roles:
            return True
        return area.lower() in [a.lower() for a in self.areas]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "roles": [r.value for r in self.roles],
            "tags": self.tags,
            "areas": self.areas,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(
            id=data["id"],
            roles=[Role(r) for r in data.get("roles", [])],
            tags=data.get("tags", []),
            areas=data.get("areas", []),
        )

    @classmethod
    def anonymous(cls) -> "User":
        """Usuário anônimo com acesso mínimo."""
        return cls(id="anonymous", roles=[Role.VIEWER], tags=[], areas=[])

    @classmethod
    def admin(cls, user_id: str) -> "User":
        """Usuário admin com acesso total."""
        return cls(id=user_id, roles=[Role.ADMIN], tags=["*"], areas=["*"])


class RBACFilter:
    """Filtro RBAC para queries de documentos."""

    def __init__(self, user: User):
        self.user = user

    def can_access_classification(self, classification: str) -> bool:
        """Verifica se usuário pode acessar classificação."""
        from models.document import Classification

        classification = Classification(classification)

        # Admin pode tudo
        if self.user.has_role(Role.ADMIN):
            return True

        # Viewer só pode público
        if Role.VIEWER in self.user.roles and not self.user.has_role(Role.ANALYST):
            return classification == Classification.PUBLIC

        # Analyst pode interno e público
        if self.user.has_role(Role.ANALYST):
            return classification in [Classification.PUBLIC, Classification.INTERNAL]

        # Outros precisam de tags específicas para confidential/restricted
        return classification in [Classification.PUBLIC, Classification.INTERNAL]

    def can_access_document(self, doc_metadata: dict) -> bool:
        """Verifica se usuário pode acessar documento baseado nos metadados."""
        # Admin pode tudo
        if self.user.has_role(Role.ADMIN):
            logger.log_rbac(self.user.id, "read", f"doc:{doc_metadata.get('id')}", True, reason="admin")
            return True

        # Verificar classificação
        classification = doc_metadata.get("classification", "internal")
        if not self.can_access_classification(classification):
            logger.log_rbac(self.user.id, "read", f"doc:{doc_metadata.get('id')}", False,
                          reason=f"classification:{classification}")
            return False

        # Verificar owner_area
        owner_area = doc_metadata.get("owner_area")
        if owner_area and not self.user.has_area_access(owner_area):
            # Verificar se tem tags que permitem acesso
            rbac_tags = doc_metadata.get("rbac_tags", [])
            if not any(self.user.has_tag(tag) for tag in rbac_tags):
                logger.log_rbac(self.user.id, "read", f"doc:{doc_metadata.get('id')}", False,
                              reason=f"area:{owner_area}")
                return False

        logger.log_rbac(self.user.id, "read", f"doc:{doc_metadata.get('id')}", True)
        return True

    def build_sql_filter(self) -> tuple[str, list]:
        """
        Constrói cláusula WHERE SQL para filtrar documentos.

        Returns:
            Tuple (sql_clause, params) para adicionar à query
        """
        if self.user.has_role(Role.ADMIN):
            return "", []

        conditions = []
        params = []

        # Filtro por classificação
        if self.user.has_role(Role.VIEWER) and not self.user.has_role(Role.ANALYST):
            conditions.append("json_extract(metadata_json, '$.classification') = ?")
            params.append("public")
        elif self.user.has_role(Role.ANALYST):
            conditions.append("json_extract(metadata_json, '$.classification') IN (?, ?)")
            params.extend(["public", "internal"])

        # Filtro por área (se usuário tem áreas definidas)
        if self.user.areas and "*" not in self.user.areas:
            area_placeholders = ", ".join(["?" for _ in self.user.areas])
            conditions.append(f"""
                (json_extract(metadata_json, '$.owner_area') IS NULL
                 OR json_extract(metadata_json, '$.owner_area') IN ({area_placeholders}))
            """)
            params.extend(self.user.areas)

        if conditions:
            return " AND " + " AND ".join(conditions), params

        return "", []


# Context var para usuário atual (similar ao logger)
from contextvars import ContextVar

current_user_var: ContextVar[Optional[User]] = ContextVar("current_user", default=None)


def set_current_user(user: User) -> None:
    """Define usuário para o contexto atual."""
    current_user_var.set(user)


def get_current_user() -> User:
    """Retorna usuário do contexto atual (ou anônimo)."""
    user = current_user_var.get()
    return user if user else User.anonymous()


def get_rbac_filter() -> RBACFilter:
    """Retorna filtro RBAC para o usuário atual."""
    return RBACFilter(get_current_user())


# Utilitários
def parse_user_from_header(auth_header: Optional[str]) -> User:
    """
    Parse usuário de header de autorização.

    Formato esperado: Bearer <base64_json>
    JSON: {"id": "user@email.com", "roles": ["analyst"], "tags": [...], "areas": [...]}
    """
    if not auth_header:
        return User.anonymous()

    try:
        import base64

        if not auth_header.startswith("Bearer "):
            return User.anonymous()

        token = auth_header[7:]  # Remove "Bearer "
        decoded = base64.b64decode(token).decode("utf-8")
        data = json.loads(decoded)
        return User.from_dict(data)
    except Exception:
        return User.anonymous()


if __name__ == "__main__":
    # Testes
    print("=== Teste de RBAC ===\n")

    # Criar usuários de teste
    admin = User.admin("admin@empresa.com")
    analyst = User(
        id="analyst@empresa.com",
        roles=[Role.ANALYST],
        tags=["ti:read", "compliance:read"],
        areas=["TI", "Compliance"]
    )
    viewer = User(
        id="viewer@empresa.com",
        roles=[Role.VIEWER],
        tags=[],
        areas=[]
    )

    # Documento de teste
    doc_metadata = {
        "id": 1,
        "classification": "confidential",
        "owner_area": "TI",
        "rbac_tags": ["ti:read"],
    }

    print("Documento: confidential, owner=TI, tags=[ti:read]\n")

    for user in [admin, analyst, viewer]:
        rbac = RBACFilter(user)
        can_access = rbac.can_access_document(doc_metadata)
        print(f"{user.id} ({user.roles[0].value}): {'ALLOWED' if can_access else 'DENIED'}")

    # Teste de SQL filter
    print("\n=== SQL Filters ===")
    for user in [admin, analyst, viewer]:
        rbac = RBACFilter(user)
        sql, params = rbac.build_sql_filter()
        print(f"\n{user.id}:")
        print(f"  SQL: {sql or '(no filter)'}")
        print(f"  Params: {params}")
