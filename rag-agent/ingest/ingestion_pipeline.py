# =============================================================================
# INGESTION PIPELINE - Pipeline Configurável de Ingestão de Documentos
# =============================================================================
# Processa documentos com chunking configurável e gera embeddings
# =============================================================================

import sys
from pathlib import Path
from typing import Optional, List
import apsw
import sqlite_vec
from fastembed import TextEmbedding
from dataclasses import dataclass

# Adicionar parent ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.chunker import Chunker, ChunkingStrategy, Chunk
from core.config import get_config
from core.logger import logger


@dataclass
class IngestionResult:
    """Resultado da ingestão de um documento."""
    doc_id: int
    doc_name: str
    chunks_created: int
    embeddings_created: int
    success: bool
    error: Optional[str] = None


class IngestionPipeline:
    """Pipeline de ingestão com chunking e embedding configuráveis."""

    def __init__(
        self,
        db_path: str,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        """
        Inicializa pipeline de ingestão.

        Args:
            db_path: Caminho do banco de dados
            chunking_strategy: Estratégia de chunking (None = usar config)
            chunk_size: Tamanho do chunk (None = usar config)
            chunk_overlap: Overlap do chunk (None = usar config)
        """
        self.db_path = db_path
        self.config = get_config()

        # Usar config se não especificado
        self.chunking_strategy = chunking_strategy or self.config.chunking_strategy
        self.chunk_size = chunk_size or self.config.chunk_size
        self.chunk_overlap = chunk_overlap or self.config.chunk_overlap

        # Criar chunker
        self.chunker = Chunker(
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
            strategy=self.chunking_strategy,
            min_chunk_size=self.config.min_chunk_size,
        )

        # Carregar modelo de embeddings
        logger.info(
            "initializing_pipeline",
            model=self.config.embedding_model.value,
            chunking=self.chunking_strategy.value,
            chunk_size=self.chunk_size,
        )
        self.model = TextEmbedding(self.config.embedding_model.value)

    def get_connection(self):
        """Cria conexão com sqlite-vec."""
        conn = apsw.Connection(self.db_path)
        conn.enableloadextension(True)
        conn.loadextension(sqlite_vec.loadable_path())
        conn.enableloadextension(False)
        return conn

    def ingest_document(
        self,
        doc_id: int,
        doc_name: str,
        content: str,
    ) -> IngestionResult:
        """
        Processa um documento: chunking + embedding + inserção no banco.

        Args:
            doc_id: ID do documento
            doc_name: Nome do documento
            content: Conteúdo do documento

        Returns:
            IngestionResult com estatísticas
        """
        try:
            # 1. Chunking
            chunks = self.chunker.chunk(content, doc_id=doc_id)

            if not chunks:
                return IngestionResult(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    chunks_created=0,
                    embeddings_created=0,
                    success=False,
                    error="No chunks created",
                )

            logger.info(
                "chunks_created",
                doc_id=doc_id,
                doc_name=doc_name,
                chunks_count=len(chunks),
                strategy=self.chunking_strategy.value,
            )

            # 2. Gerar embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = list(self.model.embed(chunk_texts))

            # 3. Inserir no banco
            conn = self.get_connection()
            cursor = conn.cursor()

            # Para cada chunk, criar um embedding
            # Nota: Neste modelo, cada documento tem UM embedding que é do documento completo
            # Se quiser múltiplos embeddings por documento (um por chunk), precisaria
            # modificar o schema do banco

            # Por enquanto, vamos usar apenas o primeiro chunk como representativo
            # ou fazer uma média dos embeddings
            if len(embeddings) == 1:
                final_embedding = embeddings[0].tolist()
            else:
                # Média dos embeddings dos chunks
                import numpy as np
                final_embedding = np.mean([e.tolist() for e in embeddings], axis=0).tolist()

            embedding_bytes = sqlite_vec.serialize_float32(final_embedding)

            # Inserir ou atualizar embedding
            cursor.execute("""
                INSERT OR REPLACE INTO vec_documentos (doc_id, embedding)
                VALUES (?, ?)
            """, (doc_id, embedding_bytes))

            conn.close()

            logger.info(
                "document_ingested",
                doc_id=doc_id,
                doc_name=doc_name,
                chunks=len(chunks),
                embeddings=1,
            )

            return IngestionResult(
                doc_id=doc_id,
                doc_name=doc_name,
                chunks_created=len(chunks),
                embeddings_created=1,
                success=True,
            )

        except Exception as e:
            logger.log_error("ingestion_failed", str(e), doc_id=doc_id, doc_name=doc_name)
            return IngestionResult(
                doc_id=doc_id,
                doc_name=doc_name,
                chunks_created=0,
                embeddings_created=0,
                success=False,
                error=str(e),
            )

    def ingest_all_documents(self) -> List[IngestionResult]:
        """
        Reindexa todos os documentos do banco.

        Returns:
            Lista de IngestionResult
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Buscar todos os documentos
        docs = []
        for row in cursor.execute("SELECT id, nome, conteudo FROM documentos"):
            docs.append((row[0], row[1], row[2]))

        conn.close()

        logger.info("starting_batch_ingestion", total_docs=len(docs))

        # Processar cada documento
        results = []
        for doc_id, doc_name, content in docs:
            if not content:
                logger.warning("skipping_empty_document", doc_id=doc_id, doc_name=doc_name)
                continue

            result = self.ingest_document(doc_id, doc_name, content)
            results.append(result)

        # Log resumo
        success_count = sum(1 for r in results if r.success)
        failed_count = len(results) - success_count
        total_chunks = sum(r.chunks_created for r in results)

        logger.info(
            "batch_ingestion_complete",
            total=len(results),
            success=success_count,
            failed=failed_count,
            total_chunks=total_chunks,
        )

        return results


if __name__ == "__main__":
    # Teste do pipeline
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline de ingestão de documentos")
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Caminho do banco de dados",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="semantic",
        choices=["fixed_size", "sentence", "paragraph", "semantic"],
        help="Estratégia de chunking",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Tamanho do chunk em tokens",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap em tokens",
    )

    args = parser.parse_args()

    # DB path
    if args.db_path:
        db_path = args.db_path
    else:
        config = get_config()
        db_path = str(config.db_path)

    print("=" * 60)
    print("🔄 PIPELINE DE INGESTÃO")
    print("=" * 60)
    print(f"Banco de dados: {db_path}")
    print(f"Estratégia: {args.strategy}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Overlap: {args.overlap}")
    print("=" * 60 + "\n")

    # Criar pipeline
    pipeline = IngestionPipeline(
        db_path=db_path,
        chunking_strategy=ChunkingStrategy(args.strategy),
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )

    # Processar todos os documentos
    results = pipeline.ingest_all_documents()

    # Mostrar resultados
    print("\n" + "=" * 60)
    print("📊 RESUMO DA INGESTÃO")
    print("=" * 60)

    for result in results:
        status = "✅" if result.success else "❌"
        print(f"{status} {result.doc_name}")
        if result.success:
            print(f"   Chunks: {result.chunks_created} | Embeddings: {result.embeddings_created}")
        else:
            print(f"   Erro: {result.error}")

    success_count = sum(1 for r in results if r.success)
    print("\n" + "=" * 60)
    print(f"Total: {len(results)} | Sucesso: {success_count} | Falhas: {len(results) - success_count}")
    print("=" * 60)
