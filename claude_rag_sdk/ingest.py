"""Ingestion engine for ClaudeRAG SDK - Document processing and indexing."""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import apsw
import sqlite_vec
from fastembed import TextEmbedding

from .options import ChunkingStrategy, EmbeddingModel


@dataclass
class IngestResult:
    """Result of document ingestion.

    Attributes:
        success: Whether ingestion succeeded
        doc_id: Database ID of the document
        chunks: Number of chunks created
        source: Source file path
        error: Error message if failed
    """
    success: bool
    doc_id: Optional[int] = None
    chunks: int = 0
    source: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "doc_id": self.doc_id,
            "chunks": self.chunks,
            "source": self.source,
            "error": self.error,
        }


@dataclass
class Document:
    """Document to be ingested.

    Attributes:
        content: Document text content
        source: Source identifier (filename, URL, etc)
        doc_type: Document type (pdf, docx, html, txt)
        metadata: Additional metadata
    """
    content: str
    source: str
    doc_type: str = "txt"
    metadata: Optional[dict] = None


class IngestEngine:
    """Document ingestion and indexing engine.

    Example:
        >>> engine = IngestEngine(db_path='rag.db')
        >>> result = await engine.add_document('/path/to/doc.pdf')
        >>> print(f"Ingested {result.chunks} chunks")

        >>> result = await engine.add_text(
        ...     "Document content here",
        ...     source="manual.txt",
        ...     metadata={"author": "John"}
        ... )
    """

    def __init__(
        self,
        db_path: str,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED,
    ):
        """Initialize ingestion engine.

        Args:
            db_path: Path to sqlite-vec database
            embedding_model: FastEmbed model name
            chunk_size: Size of chunks in tokens
            chunk_overlap: Overlap between chunks
            chunking_strategy: Strategy for chunking
        """
        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy

        # Lazy load
        self._model: Optional[TextEmbedding] = None

        # Ensure database exists
        self._ensure_database()

    @property
    def model(self) -> TextEmbedding:
        """Lazy load embedding model."""
        if self._model is None:
            self._model = TextEmbedding(self.embedding_model_name)
        return self._model

    def _get_connection(self) -> apsw.Connection:
        """Create connection with sqlite-vec loaded."""
        conn = apsw.Connection(str(self.db_path))
        conn.enableloadextension(True)
        conn.loadextension(sqlite_vec.loadable_path())
        conn.enableloadextension(False)
        return conn

    def _ensure_database(self):
        """Ensure database schema exists."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documentos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                tipo TEXT,
                conteudo TEXT,
                caminho TEXT,
                hash TEXT UNIQUE,
                metadata TEXT,
                criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Get embedding dimensions from model
        # BGE models: small=384, base=768, large=1024
        dims = 384  # default for bge-small
        if "base" in self.embedding_model_name:
            dims = 768
        elif "large" in self.embedding_model_name:
            dims = 1024

        # Create vector table
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_documentos USING vec0(
                doc_id INTEGER PRIMARY KEY,
                embedding FLOAT[{dims}]
            )
        """)

        conn.close()

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks based on strategy."""
        if self.chunking_strategy == ChunkingStrategy.FIXED:
            return self._chunk_fixed(text)
        elif self.chunking_strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_sentence(text)
        elif self.chunking_strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_paragraph(text)
        else:
            return self._chunk_fixed(text)

    def _chunk_fixed(self, text: str) -> list[str]:
        """Fixed-size chunking with overlap."""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks if chunks else [text]

    def _chunk_sentence(self, text: str) -> list[str]:
        """Sentence-based chunking."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap
                overlap_words = " ".join(current_chunk).split()[-self.chunk_overlap:]
                current_chunk = [" ".join(overlap_words)]
                current_size = len(overlap_words)

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]

    def _chunk_paragraph(self, text: str) -> list[str]:
        """Paragraph-based chunking."""
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para.split())
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks if chunks else [text]

    def _extract_text(self, file_path: Path) -> tuple[str, str]:
        """Extract text from file. Returns (content, type)."""
        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            return file_path.read_text(encoding="utf-8"), "txt"

        elif suffix == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                return text, "pdf"
            except ImportError:
                raise ImportError("pypdf required for PDF files: pip install pypdf")

        elif suffix == ".docx":
            try:
                import docx
                doc = docx.Document(str(file_path))
                text = "\n".join(para.text for para in doc.paragraphs)
                return text, "docx"
            except ImportError:
                raise ImportError("python-docx required for DOCX files: pip install python-docx")

        elif suffix in (".html", ".htm"):
            try:
                from bs4 import BeautifulSoup
                html = file_path.read_text(encoding="utf-8")
                soup = BeautifulSoup(html, "html.parser")
                return soup.get_text(separator="\n"), "html"
            except ImportError:
                raise ImportError("beautifulsoup4 required for HTML files: pip install beautifulsoup4")

        elif suffix == ".md":
            return file_path.read_text(encoding="utf-8"), "markdown"

        elif suffix == ".json":
            import json
            data = json.loads(file_path.read_text(encoding="utf-8"))
            return json.dumps(data, indent=2), "json"

        else:
            # Try to read as text
            try:
                return file_path.read_text(encoding="utf-8"), "txt"
            except UnicodeDecodeError:
                raise ValueError(f"Cannot extract text from {suffix} files")

    async def add_document(
        self,
        path: Union[str, Path],
        metadata: Optional[dict] = None,
    ) -> IngestResult:
        """Add document from file path.

        Args:
            path: Path to document file
            metadata: Optional metadata dict

        Returns:
            IngestResult with status
        """
        file_path = Path(path)
        if not file_path.exists():
            return IngestResult(
                success=False,
                source=str(path),
                error=f"File not found: {path}",
            )

        try:
            content, doc_type = self._extract_text(file_path)
            return await self.add_text(
                content=content,
                source=file_path.name,
                doc_type=doc_type,
                metadata=metadata,
                file_path=str(file_path),
            )
        except Exception as e:
            return IngestResult(
                success=False,
                source=str(path),
                error=str(e),
            )

    async def add_text(
        self,
        content: str,
        source: str,
        doc_type: str = "txt",
        metadata: Optional[dict] = None,
        file_path: Optional[str] = None,
    ) -> IngestResult:
        """Add text content directly.

        Args:
            content: Text content to index
            source: Source identifier
            doc_type: Document type
            metadata: Optional metadata
            file_path: Optional original file path

        Returns:
            IngestResult with status
        """
        if not content or not content.strip():
            return IngestResult(
                success=False,
                source=source,
                error="Empty content",
            )

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Check for duplicates
            content_hash = self._compute_hash(content)
            existing = None
            for row in cursor.execute(
                "SELECT id FROM documentos WHERE hash = ?",
                (content_hash,)
            ):
                existing = row[0]
                break

            if existing:
                return IngestResult(
                    success=True,
                    doc_id=existing,
                    chunks=0,
                    source=source,
                    error="Document already exists (duplicate)",
                )

            # Insert document
            import json
            metadata_json = json.dumps(metadata) if metadata else None

            cursor.execute("""
                INSERT INTO documentos (nome, tipo, conteudo, caminho, hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (source, doc_type, content, file_path, content_hash, metadata_json))

            doc_id = conn.last_insert_rowid()

            # Chunk and embed
            chunks = self._chunk_text(content)
            embeddings = list(self.model.embed(chunks))

            # For now, store the full document embedding (first chunk or average)
            if embeddings:
                # Use first chunk embedding as document embedding
                doc_embedding = embeddings[0].tolist()
                embedding_bytes = sqlite_vec.serialize_float32(doc_embedding)

                cursor.execute("""
                    INSERT INTO vec_documentos (doc_id, embedding)
                    VALUES (?, ?)
                """, (doc_id, embedding_bytes))

            conn.close()

            return IngestResult(
                success=True,
                doc_id=doc_id,
                chunks=len(chunks),
                source=source,
            )

        except Exception as e:
            conn.close()
            return IngestResult(
                success=False,
                source=source,
                error=str(e),
            )

    async def add_documents(
        self,
        paths: list[Union[str, Path]],
        metadata: Optional[dict] = None,
    ) -> list[IngestResult]:
        """Add multiple documents.

        Args:
            paths: List of file paths
            metadata: Metadata to apply to all documents

        Returns:
            List of IngestResult
        """
        results = []
        for path in paths:
            result = await self.add_document(path, metadata)
            results.append(result)
        return results

    async def delete_document(self, doc_id: int) -> bool:
        """Delete document by ID.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM vec_documentos WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documentos WHERE id = ?", (doc_id,))

        deleted = conn.changes() > 0
        conn.close()

        return deleted

    async def clear_all(self) -> int:
        """Delete all documents.

        Returns:
            Number of documents deleted
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM documentos")
        count = cursor.fetchone()[0]

        cursor.execute("DELETE FROM vec_documentos")
        cursor.execute("DELETE FROM documentos")

        conn.close()

        return count

    async def reindex(self) -> int:
        """Reindex all documents (regenerate embeddings).

        Returns:
            Number of documents reindexed
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get all documents
        documents = []
        for row in cursor.execute("SELECT id, conteudo FROM documentos"):
            documents.append((row[0], row[1]))

        # Clear embeddings
        cursor.execute("DELETE FROM vec_documentos")

        # Regenerate
        count = 0
        for doc_id, content in documents:
            if content:
                chunks = self._chunk_text(content)
                embeddings = list(self.model.embed(chunks))

                if embeddings:
                    doc_embedding = embeddings[0].tolist()
                    embedding_bytes = sqlite_vec.serialize_float32(doc_embedding)

                    cursor.execute("""
                        INSERT INTO vec_documentos (doc_id, embedding)
                        VALUES (?, ?)
                    """, (doc_id, embedding_bytes))

                    count += 1

        conn.close()

        return count

    @property
    def stats(self) -> dict:
        """Get ingestion statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        total_docs = 0
        for r in cursor.execute("SELECT COUNT(*) FROM documentos"):
            total_docs = r[0]

        total_embeddings = 0
        for r in cursor.execute("SELECT COUNT(*) FROM vec_documentos"):
            total_embeddings = r[0]

        total_size = 0
        for r in cursor.execute("SELECT SUM(LENGTH(conteudo)) FROM documentos"):
            total_size = r[0] or 0

        conn.close()

        return {
            "total_documents": total_docs,
            "total_embeddings": total_embeddings,
            "total_size_bytes": total_size,
            "status": "ok" if total_docs == total_embeddings else "incompleto",
        }
