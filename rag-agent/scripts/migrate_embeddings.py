#!/usr/bin/env python3
# =============================================================================
# MIGRATE EMBEDDINGS - Script de Migração de Modelos de Embedding
# =============================================================================
# Migra embeddings de um modelo para outro (ex: bge-small → bge-large)
# Com backup, rollback e progress tracking
# =============================================================================

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import apsw
import sqlite_vec
from fastembed import TextEmbedding

# Adicionar parent ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_config, EmbeddingModel
from core.logger import logger


class EmbeddingMigrator:
    """Migra embeddings entre modelos diferentes."""

    def __init__(
        self,
        db_path: str,
        source_model: EmbeddingModel,
        target_model: EmbeddingModel,
        batch_size: int = 10,
    ):
        self.db_path = db_path
        self.source_model = source_model
        self.target_model = target_model
        self.batch_size = batch_size

        # Inicializar modelo target
        print(f"📦 Carregando modelo {target_model.value}...")
        self.model = TextEmbedding(target_model.value)
        print(f"✅ Modelo carregado ({target_model.dimensions} dimensões)")

        # Stats
        self.stats = {
            "total_docs": 0,
            "processed": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None,
        }

    def get_connection(self):
        """Cria conexão com sqlite-vec."""
        conn = apsw.Connection(self.db_path)
        conn.enableloadextension(True)
        conn.loadextension(sqlite_vec.loadable_path())
        conn.enableloadextension(False)
        return conn

    def backup_table(self):
        """Faz backup da tabela vec_documentos."""
        print("\n🗄️  Fazendo backup da tabela vec_documentos...")
        conn = self.get_connection()
        cursor = conn.cursor()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_table = f"vec_documentos_backup_{timestamp}"

        try:
            # Criar tabela de backup
            cursor.execute(f"""
                CREATE VIRTUAL TABLE {backup_table} USING vec0(
                    doc_id INTEGER PRIMARY KEY,
                    embedding float[{self.source_model.dimensions}]
                )
            """)

            # Copiar dados
            cursor.execute(f"""
                INSERT INTO {backup_table} (doc_id, embedding)
                SELECT doc_id, embedding FROM vec_documentos
            """)

            # Contar registros
            backup_count = 0
            for row in cursor.execute(f"SELECT COUNT(*) FROM {backup_table}"):
                backup_count = row[0]

            print(f"✅ Backup criado: {backup_table} ({backup_count:,} registros)")

        except Exception as e:
            print(f"❌ Erro ao fazer backup: {e}")
            raise
        finally:
            conn.close()

        return backup_table

    def create_new_table(self):
        """Cria nova tabela vec_documentos_v2 com dimensões corretas."""
        print(f"\n🔧 Criando nova tabela (dimensões: {self.target_model.dimensions})...")
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Dropar se já existir
            cursor.execute("DROP TABLE IF EXISTS vec_documentos_v2")

            # Criar nova tabela
            cursor.execute(f"""
                CREATE VIRTUAL TABLE vec_documentos_v2 USING vec0(
                    doc_id INTEGER PRIMARY KEY,
                    embedding float[{self.target_model.dimensions}]
                )
            """)

            print("✅ Tabela vec_documentos_v2 criada")

        except Exception as e:
            print(f"❌ Erro ao criar tabela: {e}")
            raise
        finally:
            conn.close()

    def migrate_embeddings(self):
        """Migra embeddings para novo modelo."""
        print(f"\n🚀 Iniciando migração de embeddings...")
        print(f"Modelo origem: {self.source_model.short_name} ({self.source_model.dimensions}D)")
        print(f"Modelo destino: {self.target_model.short_name} ({self.target_model.dimensions}D)")
        print(f"Batch size: {self.batch_size}\n")

        self.stats["start_time"] = time.time()

        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Contar documentos
            for row in cursor.execute("SELECT COUNT(*) FROM documentos"):
                self.stats["total_docs"] = row[0]

            print(f"📚 Total de documentos: {self.stats['total_docs']:,}\n")

            # Buscar documentos
            docs = []
            for row in cursor.execute("SELECT id, conteudo FROM documentos WHERE conteudo IS NOT NULL"):
                docs.append((row[0], row[1]))

            # Processar em batches
            for i in range(0, len(docs), self.batch_size):
                batch = docs[i:i + self.batch_size]
                self._process_batch(cursor, batch)

                # Progress
                self.stats["processed"] = min(i + self.batch_size, len(docs))
                self._print_progress()

            self.stats["end_time"] = time.time()

        except Exception as e:
            print(f"\n❌ Erro durante migração: {e}")
            logger.log_error("MigrationFailed", str(e))
            raise
        finally:
            conn.close()

        self._print_summary()

    def _process_batch(self, cursor, batch: list[tuple[int, str]]):
        """Processa um batch de documentos."""
        doc_ids = [doc_id for doc_id, _ in batch]
        texts = [text for _, text in batch]

        try:
            # Gerar embeddings
            embeddings = list(self.model.embed(texts))

            # Inserir no banco
            for doc_id, embedding in zip(doc_ids, embeddings):
                embedding_bytes = sqlite_vec.serialize_float32(embedding.tolist())
                cursor.execute("""
                    INSERT OR REPLACE INTO vec_documentos_v2 (doc_id, embedding)
                    VALUES (?, ?)
                """, (doc_id, embedding_bytes))

        except Exception as e:
            print(f"\n⚠️  Erro ao processar batch: {e}")
            self.stats["failed"] += len(batch)
            logger.log_error("BatchProcessingFailed", str(e), batch_size=len(batch))

    def _print_progress(self):
        """Imprime progresso da migração."""
        total = self.stats["total_docs"]
        processed = self.stats["processed"]
        failed = self.stats["failed"]
        pct = (processed / total * 100) if total > 0 else 0

        # Calcular ETA
        elapsed = time.time() - self.stats["start_time"]
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0

        print(f"\r⏳ Progresso: {processed:,}/{total:,} ({pct:.1f}%) | "
              f"Falhas: {failed} | "
              f"Taxa: {rate:.1f} docs/s | "
              f"ETA: {eta:.0f}s", end="")

    def _print_summary(self):
        """Imprime resumo da migração."""
        print("\n\n" + "=" * 60)
        print("📊 RESUMO DA MIGRAÇÃO")
        print("=" * 60)

        total = self.stats["total_docs"]
        processed = self.stats["processed"]
        failed = self.stats["failed"]
        elapsed = self.stats["end_time"] - self.stats["start_time"]

        print(f"Total de documentos: {total:,}")
        print(f"Processados com sucesso: {processed - failed:,}")
        print(f"Falharam: {failed}")
        print(f"Taxa de sucesso: {((processed - failed) / total * 100):.1f}%")
        print(f"Tempo total: {elapsed:.2f}s ({elapsed / 60:.2f} minutos)")
        print(f"Taxa média: {processed / elapsed:.2f} docs/s")
        print("=" * 60)

    def swap_tables(self):
        """Troca vec_documentos por vec_documentos_v2 (atomic swap)."""
        print("\n🔄 Realizando swap das tabelas...")
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Renomear original para _old
            cursor.execute("DROP TABLE IF EXISTS vec_documentos_old")
            cursor.execute("ALTER TABLE vec_documentos RENAME TO vec_documentos_old")

            # Renomear v2 para vec_documentos
            cursor.execute("ALTER TABLE vec_documentos_v2 RENAME TO vec_documentos")

            print("✅ Swap realizado com sucesso!")
            print("   vec_documentos → vec_documentos_old")
            print("   vec_documentos_v2 → vec_documentos")

        except Exception as e:
            print(f"❌ Erro ao fazer swap: {e}")
            raise
        finally:
            conn.close()

    def verify_migration(self):
        """Verifica se migração foi bem sucedida."""
        print("\n🔍 Verificando migração...")
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Contar registros em vec_documentos
            new_count = 0
            for row in cursor.execute("SELECT COUNT(*) FROM vec_documentos"):
                new_count = row[0]

            # Contar documentos
            total_docs = 0
            for row in cursor.execute("SELECT COUNT(*) FROM documentos"):
                total_docs = row[0]

            print(f"Documentos na tabela documentos: {total_docs:,}")
            print(f"Embeddings na tabela vec_documentos: {new_count:,}")

            if new_count == total_docs:
                print("✅ Migração verificada com sucesso!")
                return True
            else:
                print(f"⚠️  Contagem divergente: {new_count} vs {total_docs}")
                return False

        except Exception as e:
            print(f"❌ Erro ao verificar: {e}")
            return False
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Migrar embeddings entre modelos")
    parser.add_argument(
        "--source",
        type=str,
        default="bge-small",
        choices=["bge-small", "bge-base", "bge-large"],
        help="Modelo de origem (padrão: bge-small)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="bge-large",
        choices=["bge-small", "bge-base", "bge-large"],
        help="Modelo de destino (padrão: bge-large)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Caminho do banco de dados (padrão: config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Tamanho do batch para processamento (padrão: 10)",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Pular backup (NÃO RECOMENDADO)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simular migração sem fazer alterações",
    )

    args = parser.parse_args()

    # Mapear modelos
    model_map = {
        "bge-small": EmbeddingModel.BGE_SMALL,
        "bge-base": EmbeddingModel.BGE_BASE,
        "bge-large": EmbeddingModel.BGE_LARGE,
    }

    source_model = model_map[args.source]
    target_model = model_map[args.target]

    if source_model == target_model:
        print("❌ Modelo de origem e destino são iguais!")
        return

    # DB path
    if args.db_path:
        db_path = args.db_path
    else:
        config = get_config()
        db_path = str(config.db_path)

    print("=" * 60)
    print("🔄 MIGRAÇÃO DE EMBEDDINGS")
    print("=" * 60)
    print(f"Banco de dados: {db_path}")
    print(f"Modelo origem: {source_model.value} ({source_model.dimensions}D)")
    print(f"Modelo destino: {target_model.value} ({target_model.dimensions}D)")
    print(f"Batch size: {args.batch_size}")

    if args.dry_run:
        print("\n⚠️  DRY RUN - Nenhuma alteração será feita")

    if args.skip_backup:
        print("\n⚠️  ATENÇÃO: Backup será PULADO!")

    print("\n" + "=" * 60)

    # Confirmar
    if not args.dry_run:
        confirm = input("\n⚠️  Esta operação irá modificar o banco de dados. Continuar? (sim/não): ")
        if confirm.lower() != "sim":
            print("❌ Operação cancelada pelo usuário.")
            return

    # Criar migrator
    migrator = EmbeddingMigrator(
        db_path=db_path,
        source_model=source_model,
        target_model=target_model,
        batch_size=args.batch_size,
    )

    try:
        # Backup
        if not args.skip_backup and not args.dry_run:
            backup_table = migrator.backup_table()

        if args.dry_run:
            print("\n✅ Dry run completo - nenhuma alteração foi feita")
            return

        # Criar nova tabela
        migrator.create_new_table()

        # Migrar embeddings
        migrator.migrate_embeddings()

        # Swap tables
        migrator.swap_tables()

        # Verificar
        if migrator.verify_migration():
            print("\n🎉 Migração completa com sucesso!")
            print("\n💡 Para reverter, use:")
            print(f"   ALTER TABLE vec_documentos RENAME TO vec_documentos_failed;")
            print(f"   ALTER TABLE vec_documentos_old RENAME TO vec_documentos;")
        else:
            print("\n⚠️  Migração completa mas verificação falhou. Verifique manualmente.")

    except Exception as e:
        print(f"\n❌ Migração falhou: {e}")
        print("\n💡 Para reverter mudanças, use o backup criado.")
        sys.exit(1)


if __name__ == "__main__":
    main()
