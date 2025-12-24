# =============================================================================
# SECURITY - CORS e Headers de Segurança
# =============================================================================
# Configuração de CORS restritivo e headers de segurança para produção
# =============================================================================

from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class CORSConfig:
    """Configuração de CORS."""
    allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    allowed_methods: list[str] = field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
    allowed_headers: list[str] = field(default_factory=lambda: ["Content-Type", "Authorization", "X-Request-ID"])
    expose_headers: list[str] = field(default_factory=lambda: ["X-Request-ID", "X-RateLimit-Remaining"])
    allow_credentials: bool = False
    max_age: int = 86400  # 24 horas


class CORSMiddleware:
    """Middleware de CORS restritivo."""

    def __init__(self, config: Optional[CORSConfig] = None):
        self.config = config or CORSConfig()
        self._origin_patterns = self._compile_patterns()

    def _compile_patterns(self) -> list[re.Pattern]:
        """Compila padrões de origem para matching."""
        patterns = []
        for origin in self.config.allowed_origins:
            # Converter wildcards para regex
            if "*" in origin:
                pattern = origin.replace(".", r"\.").replace("*", ".*")
                patterns.append(re.compile(f"^{pattern}$"))
            else:
                patterns.append(re.compile(f"^{re.escape(origin)}$"))
        return patterns

    def is_origin_allowed(self, origin: str) -> bool:
        """Verifica se origem é permitida."""
        if not origin:
            return False
        for pattern in self._origin_patterns:
            if pattern.match(origin):
                return True
        return False

    def get_cors_headers(self, origin: str) -> dict[str, str]:
        """Retorna headers CORS para a origem."""
        headers = {}

        if self.is_origin_allowed(origin):
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Methods"] = ", ".join(self.config.allowed_methods)
            headers["Access-Control-Allow-Headers"] = ", ".join(self.config.allowed_headers)
            headers["Access-Control-Expose-Headers"] = ", ".join(self.config.expose_headers)
            headers["Access-Control-Max-Age"] = str(self.config.max_age)

            if self.config.allow_credentials:
                headers["Access-Control-Allow-Credentials"] = "true"

        return headers

    def handle_preflight(self, origin: str, method: str) -> tuple[int, dict[str, str]]:
        """
        Processa requisição OPTIONS (preflight).

        Returns:
            (status_code, headers)
        """
        if not self.is_origin_allowed(origin):
            return 403, {"Content-Type": "text/plain"}

        if method not in self.config.allowed_methods:
            return 405, {"Content-Type": "text/plain"}

        return 204, self.get_cors_headers(origin)


@dataclass
class SecurityHeaders:
    """Headers de segurança para produção."""

    # Content Security Policy
    csp: str = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"

    # Prevenir clickjacking
    x_frame_options: str = "DENY"

    # Prevenir MIME sniffing
    x_content_type_options: str = "nosniff"

    # XSS Protection
    x_xss_protection: str = "1; mode=block"

    # Referrer Policy
    referrer_policy: str = "strict-origin-when-cross-origin"

    # HSTS (apenas para HTTPS)
    hsts: str = "max-age=31536000; includeSubDomains"

    # Permissions Policy
    permissions_policy: str = "geolocation=(), microphone=(), camera=()"

    def to_dict(self) -> dict[str, str]:
        """Retorna headers como dicionário."""
        return {
            "Content-Security-Policy": self.csp,
            "X-Frame-Options": self.x_frame_options,
            "X-Content-Type-Options": self.x_content_type_options,
            "X-XSS-Protection": self.x_xss_protection,
            "Referrer-Policy": self.referrer_policy,
            "Strict-Transport-Security": self.hsts,
            "Permissions-Policy": self.permissions_policy,
        }


def get_security_headers(include_hsts: bool = False) -> dict[str, str]:
    """
    Retorna headers de segurança padrão.

    Args:
        include_hsts: Incluir HSTS (apenas para HTTPS)
    """
    headers = SecurityHeaders()
    result = headers.to_dict()

    if not include_hsts:
        del result["Strict-Transport-Security"]

    return result


# Configurações predefinidas
CORS_DEVELOPMENT = CORSConfig(
    allowed_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
)

CORS_PRODUCTION = CORSConfig(
    allowed_origins=[],  # Definir origens específicas
    allowed_methods=["GET", "POST"],
    allow_credentials=False,
)


if __name__ == "__main__":
    print("=== Teste de CORS ===\n")

    cors = CORSMiddleware(CORS_DEVELOPMENT)

    test_origins = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:5000",
        "http://evil.com",
        "https://attacker.net",
    ]

    for origin in test_origins:
        allowed = cors.is_origin_allowed(origin)
        print(f"  {origin}: {'✓ Allowed' if allowed else '✗ Blocked'}")

    print("\n=== Security Headers ===")
    for k, v in get_security_headers().items():
        print(f"  {k}: {v[:50]}...")
