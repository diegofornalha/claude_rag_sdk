"""File watcher for automatic RAG reindexing using Watchdog."""

import subprocess
import threading
from pathlib import Path
from typing import Optional

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer


class BackendFileHandler(FileSystemEventHandler):
    """Handler para mudanças em arquivos Python do backend."""

    def __init__(self, ingest_script_path: Path, cooldown_seconds: int = 5):
        """
        Args:
            ingest_script_path: Caminho para o script ingest_backend.py
            cooldown_seconds: Tempo mínimo entre reingestões (evita spam)
        """
        super().__init__()
        self.ingest_script = ingest_script_path
        self.cooldown = cooldown_seconds
        self.last_run = 0
        self.pending_files = set()
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def on_modified(self, event):
        """Chamado quando arquivo é modificado."""
        if isinstance(event, (FileModifiedEvent, FileCreatedEvent)):
            self._handle_change(event.src_path)

    def on_created(self, event):
        """Chamado quando arquivo é criado."""
        if isinstance(event, FileCreatedEvent):
            self._handle_change(event.src_path)

    def _handle_change(self, file_path: str):
        """Processa mudança de arquivo."""
        path = Path(file_path)

        # Ignorar arquivos que não são Python
        if path.suffix != ".py":
            return

        # Ignorar __pycache__, .pyc, testes, etc
        if any(part.startswith((".", "__pycache__")) for part in path.parts):
            return

        # Ignorar scripts de ingestão (evitar loop)
        if "ingest" in path.name.lower() or "scripts" in str(path):
            return

        print(f"[WATCHDOG] Mudança detectada: {path.name}")

        with self._lock:
            self.pending_files.add(str(path))

            # Cancelar timer anterior se existir
            if self._timer and self._timer.is_alive():
                self._timer.cancel()

            # Agendar reingestão após cooldown
            self._timer = threading.Timer(self.cooldown, self._run_ingest)
            self._timer.start()

    def _run_ingest(self):
        """Executa script de ingestão."""
        with self._lock:
            if not self.pending_files:
                return

            files_changed = len(self.pending_files)
            self.pending_files.clear()

        print(
            f"[WATCHDOG] Executando reingestão automática ({files_changed} arquivo(s) modificado(s))..."
        )

        try:
            result = subprocess.run(
                ["python", str(self.ingest_script)],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.ingest_script.parent.parent),
            )

            if result.returncode == 0:
                # Extrair resumo do output
                output = result.stdout
                if "Nenhum arquivo modificado" in output:
                    print("[WATCHDOG] Base já estava atualizada")
                else:
                    # Tentar extrair número de arquivos processados
                    import re

                    match = re.search(r"Novos/Atualizados:\s*(\d+)", output)
                    if match:
                        count = match.group(1)
                        print(
                            f"[WATCHDOG] ✓ Reingestão concluída: {count} arquivo(s) atualizado(s)"
                        )
                    else:
                        print("[WATCHDOG] ✓ Reingestão concluída")
            else:
                print(f"[WATCHDOG] ✗ Erro na reingestão: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            print("[WATCHDOG] ✗ Timeout na reingestão (5 min)")
        except Exception as e:
            print(f"[WATCHDOG] ✗ Erro ao executar reingestão: {e}")


class FileWatcherService:
    """Serviço de monitoramento de arquivos."""

    def __init__(self, backend_path: Path, ingest_script_path: Path):
        self.backend_path = backend_path
        self.ingest_script = ingest_script_path
        self.observer: Optional[Observer] = None
        self.handler: Optional[BackendFileHandler] = None
        self._enabled = False

    def start(self):
        """Inicia o monitoramento."""
        if self._enabled:
            print("[WATCHDOG] Já está ativo")
            return

        if not self.backend_path.exists():
            raise FileNotFoundError(f"Backend path not found: {self.backend_path}")

        if not self.ingest_script.exists():
            raise FileNotFoundError(f"Ingest script not found: {self.ingest_script}")

        self.handler = BackendFileHandler(ingest_script_path=self.ingest_script, cooldown_seconds=5)

        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.backend_path), recursive=True)
        self.observer.start()
        self._enabled = True

        print(f"[WATCHDOG] Monitoramento ativo em: {self.backend_path}")

    def stop(self):
        """Para o monitoramento."""
        if not self._enabled:
            print("[WATCHDOG] Já está inativo")
            return

        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.observer = None

        self.handler = None
        self._enabled = False

        print("[WATCHDOG] Monitoramento desativado")

    def is_active(self) -> bool:
        """Verifica se está ativo."""
        return self._enabled and self.observer is not None and self.observer.is_alive()

    def get_status(self) -> dict:
        """Retorna status do watcher."""
        return {
            "enabled": self._enabled,
            "active": self.is_active(),
            "watching_path": str(self.backend_path),
            "cooldown_seconds": self.handler.cooldown if self.handler else 5,
            "pending_files": len(self.handler.pending_files) if self.handler else 0,
        }


# Global instance
_watcher: Optional[FileWatcherService] = None


def get_watcher() -> FileWatcherService:
    """Obtém instância global do watcher."""
    global _watcher
    if _watcher is None:
        backend_path = Path.cwd()
        ingest_script = backend_path / "scripts" / "ingest_backend.py"
        _watcher = FileWatcherService(backend_path, ingest_script)
    return _watcher
