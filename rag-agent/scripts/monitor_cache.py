#!/usr/bin/env python3
# =============================================================================
# MONITOR CACHE - Dashboard em tempo real de hit rate do cache
# =============================================================================
# Monitora métricas de cache em tempo real e alerta se hit rate baixo
# =============================================================================

import time
import sys
from pathlib import Path

# Adicionar parent ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cache import get_embedding_cache, get_response_cache


def format_percentage(value: float) -> str:
    """Formata percentual com cor."""
    pct = value * 100
    if pct >= 70:
        color = "\033[92m"  # Verde
    elif pct >= 50:
        color = "\033[93m"  # Amarelo
    else:
        color = "\033[91m"  # Vermelho
    reset = "\033[0m"
    return f"{color}{pct:.1f}%{reset}"


def format_size_mb(size_bytes: int) -> str:
    """Formata tamanho em MB."""
    return f"{size_bytes / 1024 / 1024:.2f} MB"


def clear_screen():
    """Limpa a tela."""
    print("\033[2J\033[H", end="")


def print_cache_stats(cache_name: str, stats):
    """Imprime estatísticas de um cache."""
    print(f"\n{cache_name}:")
    print(f"  Hits: {stats.hits:,} | Misses: {stats.misses:,} | Total: {stats.hits + stats.misses:,}")
    print(f"  Hit Rate: {format_percentage(stats.hit_rate)}")
    print(f"  Size: {stats.size:,} entries (max: {stats.max_size:,})")
    print(f"  Memory: {format_size_mb(stats.memory_bytes)}")
    print(f"  Evictions: {stats.evictions:,}")


def monitor_cache(interval: int = 5):
    """
    Monitora cache em tempo real.

    Args:
        interval: Intervalo de atualização em segundos
    """
    emb_cache = get_embedding_cache()
    resp_cache = get_response_cache()

    print("🔍 Monitorando cache do RAG Agent...")
    print(f"Intervalo de atualização: {interval}s")
    print("Pressione Ctrl+C para sair\n")

    try:
        while True:
            clear_screen()

            print("=" * 60)
            print("📊 DASHBOARD DE CACHE - RAG AGENT")
            print(f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)

            # Stats de embedding cache
            emb_stats = emb_cache.stats
            print_cache_stats("🔤 EMBEDDING CACHE", emb_stats)

            # Stats de response cache
            resp_stats = resp_cache.stats
            print_cache_stats("📝 RESPONSE CACHE", resp_stats)

            # Alertas
            print("\n" + "=" * 60)
            alerts = []

            if emb_stats.hit_rate < 0.5 and (emb_stats.hits + emb_stats.misses) > 10:
                alerts.append("⚠️  Embedding cache hit rate baixo (<50%)")

            if resp_stats.hit_rate < 0.5 and (resp_stats.hits + resp_stats.misses) > 10:
                alerts.append("⚠️  Response cache hit rate baixo (<50%)")

            if emb_stats.size >= emb_stats.max_size * 0.9:
                alerts.append("⚠️  Embedding cache quase cheio (>90%)")

            if resp_stats.size >= resp_stats.max_size * 0.9:
                alerts.append("⚠️  Response cache quase cheio (>90%)")

            if alerts:
                print("🚨 ALERTAS:")
                for alert in alerts:
                    print(f"  {alert}")
            else:
                print("✅ Tudo OK - Sem alertas")

            print("=" * 60)
            print(f"\nPróxima atualização em {interval}s... (Ctrl+C para sair)")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n👋 Monitoramento encerrado.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor de cache do RAG Agent")
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=5,
        help="Intervalo de atualização em segundos (padrão: 5)",
    )

    args = parser.parse_args()

    monitor_cache(args.interval)


if __name__ == "__main__":
    main()
