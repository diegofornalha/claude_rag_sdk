# =============================================================================
# CHUNKER - Chunking com Overlap para RAG
# =============================================================================
# Divide documentos em chunks com overlap para melhor retrieval
# =============================================================================

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional


class ChunkingStrategy(str, Enum):
    """Estratégias de chunking disponíveis."""
    FIXED_SIZE = "fixed_size"           # Tamanho fixo em tokens
    SENTENCE = "sentence"               # Por sentenças
    PARAGRAPH = "paragraph"             # Por parágrafos
    SEMANTIC = "semantic"               # Por quebras semânticas (headers, etc)


@dataclass
class Chunk:
    """Representa um chunk de documento."""
    text: str
    index: int                          # Índice do chunk no documento
    start_char: int                     # Posição inicial no texto original
    end_char: int                       # Posição final no texto original
    token_count: int                    # Número estimado de tokens
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


class Chunker:
    """Chunker com suporte a overlap."""

    def __init__(
        self,
        chunk_size: int = 500,          # Tokens por chunk
        overlap: int = 50,              # Tokens de overlap
        strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        min_chunk_size: int = 100,      # Mínimo de tokens para criar chunk
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size

        # Estimativa: 1 token ≈ 4 caracteres
        self._chars_per_token = 4

    def chunk(self, text: str, doc_id: Optional[int] = None) -> list[Chunk]:
        """
        Divide texto em chunks com overlap.

        Args:
            text: Texto a ser dividido
            doc_id: ID do documento (opcional, para metadata)

        Returns:
            Lista de Chunks
        """
        if not text or not text.strip():
            return []

        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(text, doc_id)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(text, doc_id)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(text, doc_id)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(text, doc_id)
        else:
            return self._chunk_fixed_size(text, doc_id)

    def _estimate_tokens(self, text: str) -> int:
        """Estima número de tokens no texto."""
        return len(text) // self._chars_per_token

    def _chunk_fixed_size(self, text: str, doc_id: Optional[int] = None) -> list[Chunk]:
        """Chunking por tamanho fixo com overlap."""
        chunks = []
        chunk_chars = self.chunk_size * self._chars_per_token
        overlap_chars = self.overlap * self._chars_per_token

        start = 0
        index = 0

        while start < len(text):
            end = min(start + chunk_chars, len(text))

            # Tentar quebrar em espaço/pontuação
            if end < len(text):
                # Procurar último espaço antes do limite
                last_space = text.rfind(' ', start, end)
                if last_space > start + chunk_chars // 2:
                    end = last_space

            chunk_text = text[start:end].strip()

            if chunk_text and self._estimate_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    index=index,
                    start_char=start,
                    end_char=end,
                    token_count=self._estimate_tokens(chunk_text),
                    metadata={"doc_id": doc_id, "strategy": "fixed_size"}
                ))
                index += 1
                # Próximo chunk começa com overlap
                new_start = end - overlap_chars
            else:
                # Chunk muito pequeno - avançar sem overlap para evitar loop infinito
                new_start = end

            # Garantir que sempre avançamos pelo menos 1 caractere
            start = max(new_start, start + 1)

            if start >= len(text) - self.min_chunk_size * self._chars_per_token:
                break

        return chunks

    def _chunk_by_sentence(self, text: str, doc_id: Optional[int] = None) -> list[Chunk]:
        """Chunking por sentenças com overlap."""
        # Regex para detectar fim de sentença
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0
        start_char = 0
        index = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Criar chunk atual
                chunk_text = ' '.join(current_chunk)
                end_char = start_char + len(chunk_text)

                chunks.append(Chunk(
                    text=chunk_text,
                    index=index,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=current_tokens,
                    metadata={"doc_id": doc_id, "strategy": "sentence"}
                ))
                index += 1

                # Overlap: manter últimas sentenças
                overlap_tokens = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_tokens = self._estimate_tokens(s)
                    if overlap_tokens + s_tokens <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_chunk = overlap_sentences
                current_tokens = overlap_tokens
                start_char = text.find(current_chunk[0], start_char) if current_chunk else end_char

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Último chunk
        if current_chunk and current_tokens >= self.min_chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                index=index,
                start_char=start_char,
                end_char=len(text),
                token_count=current_tokens,
                metadata={"doc_id": doc_id, "strategy": "sentence"}
            ))

        return chunks

    def _chunk_by_paragraph(self, text: str, doc_id: Optional[int] = None) -> list[Chunk]:
        """Chunking por parágrafos com overlap."""
        paragraphs = re.split(r'\n\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0
        start_char = 0
        index = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)

            # Se parágrafo único é muito grande, usar fixed_size
            if para_tokens > self.chunk_size:
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        index=index,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                        token_count=current_tokens,
                        metadata={"doc_id": doc_id, "strategy": "paragraph"}
                    ))
                    index += 1
                    current_chunk = []
                    current_tokens = 0

                # Dividir parágrafo grande
                sub_chunks = self._chunk_fixed_size(para, doc_id)
                for sc in sub_chunks:
                    sc.index = index
                    sc.metadata["strategy"] = "paragraph_split"
                    chunks.append(sc)
                    index += 1
                continue

            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                end_char = start_char + len(chunk_text)

                chunks.append(Chunk(
                    text=chunk_text,
                    index=index,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=current_tokens,
                    metadata={"doc_id": doc_id, "strategy": "paragraph"}
                ))
                index += 1

                # Overlap: último parágrafo se couber
                if self._estimate_tokens(current_chunk[-1]) <= self.overlap:
                    current_chunk = [current_chunk[-1]]
                    current_tokens = self._estimate_tokens(current_chunk[0])
                    start_char = text.find(current_chunk[0], start_char)
                else:
                    current_chunk = []
                    current_tokens = 0
                    start_char = end_char

            current_chunk.append(para)
            current_tokens += para_tokens

        # Último chunk
        if current_chunk and current_tokens >= self.min_chunk_size:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                index=index,
                start_char=start_char,
                end_char=len(text),
                token_count=current_tokens,
                metadata={"doc_id": doc_id, "strategy": "paragraph"}
            ))

        return chunks

    def _chunk_semantic(self, text: str, doc_id: Optional[int] = None) -> list[Chunk]:
        """Chunking semântico por headers e seções."""
        # Detectar headers markdown ou seções
        header_pattern = r'^(#{1,6}\s+.+|[A-Z][A-Za-z\s]+:?\s*$)'
        lines = text.split('\n')

        sections = []
        current_section = []
        current_header = None

        for line in lines:
            if re.match(header_pattern, line.strip(), re.MULTILINE):
                if current_section:
                    sections.append((current_header, '\n'.join(current_section)))
                current_header = line.strip()
                current_section = []
            else:
                current_section.append(line)

        if current_section:
            sections.append((current_header, '\n'.join(current_section)))

        # Converter seções em chunks
        chunks = []
        index = 0
        char_pos = 0

        for header, content in sections:
            full_text = f"{header}\n{content}" if header else content
            full_text = full_text.strip()

            if not full_text:
                continue

            section_length = len(full_text)
            tokens = self._estimate_tokens(full_text)
            section_produced_chunks = False

            if tokens > self.chunk_size:
                # Seção muito grande, subdividir
                sub_chunks = self._chunk_by_paragraph(full_text, doc_id)
                for sc in sub_chunks:
                    sc.index = index
                    sc.start_char += char_pos
                    sc.end_char += char_pos
                    sc.metadata["header"] = header
                    chunks.append(sc)
                    index += 1
                section_produced_chunks = len(sub_chunks) > 0
            elif tokens >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=full_text,
                    index=index,
                    start_char=char_pos,
                    end_char=char_pos + section_length,
                    token_count=tokens,
                    metadata={"doc_id": doc_id, "strategy": "semantic", "header": header}
                ))
                index += 1
                section_produced_chunks = True

            # Só avançar char_pos se a seção produziu chunks
            if section_produced_chunks:
                char_pos += section_length + 2  # +2 para \n\n entre seções

        return chunks


def reindex_documents(db_path: str, chunker: Chunker) -> dict:
    """
    Reindexa documentos existentes com nova estratégia de chunking.

    Args:
        db_path: Caminho do banco de dados
        chunker: Instância do Chunker

    Returns:
        Estatísticas de reindexação
    """
    import apsw
    import sqlite_vec
    from fastembed import TextEmbedding
    import sys
    from pathlib import Path

    # Importar config
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import get_config

    conn = apsw.Connection(db_path)
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    cursor = conn.cursor()

    # Carregar modelo de embeddings da config
    config = get_config()
    model = TextEmbedding(config.embedding_model.value)

    stats = {"docs_processed": 0, "chunks_created": 0, "embeddings_created": 0}

    # Buscar todos os documentos
    docs = list(cursor.execute("SELECT id, nome, conteudo FROM documentos"))

    for doc_id, nome, conteudo in docs:
        if not conteudo:
            continue

        # Criar chunks
        chunks = chunker.chunk(conteudo, doc_id)
        stats["chunks_created"] += len(chunks)

        # Gerar embeddings para cada chunk
        for chunk in chunks:
            embeddings = list(model.embed([chunk.text]))
            embedding_bytes = sqlite_vec.serialize_float32(embeddings[0].tolist())

            # Inserir ou atualizar embedding
            # Nota: Em produção, você criaria uma tabela separada para chunks
            stats["embeddings_created"] += 1

        stats["docs_processed"] += 1

    conn.close()
    return stats


if __name__ == "__main__":
    # Teste do chunker
    sample_text = """
# Política de Uso de IA

## Introdução

Esta política estabelece os princípios fundamentais para uso de Inteligência Artificial na organização. Todos os colaboradores devem seguir estas diretrizes ao desenvolver ou utilizar sistemas de IA.

## Princípios Obrigatórios

1. **Transparência**: Todos os sistemas de IA devem ser documentados e explicáveis.
2. **Responsabilidade**: Deve haver um responsável claro por cada sistema de IA.
3. **Privacidade**: Dados pessoais devem ser protegidos conforme LGPD.
4. **Equidade**: Sistemas não devem discriminar grupos ou indivíduos.

## Governança

A área de TI é responsável pela governança de IA, incluindo:
- Aprovação de novos projetos de IA
- Auditoria periódica de sistemas existentes
- Treinamento de colaboradores

## Métricas de Monitoramento

Para cada sistema de IA em produção, devem ser monitoradas:
- Latência de resposta (p95 < 3s)
- Taxa de erro (< 1%)
- Satisfação do usuário (> 4.0/5.0)
- Custo por requisição

## Conclusão

O cumprimento desta política é obrigatório e será auditado trimestralmente.
"""

    print("=== Teste de Chunking ===\n")

    # Testar diferentes estratégias
    strategies = [
        (ChunkingStrategy.FIXED_SIZE, 200, 50),
        (ChunkingStrategy.SENTENCE, 200, 30),
        (ChunkingStrategy.PARAGRAPH, 300, 50),
        (ChunkingStrategy.SEMANTIC, 300, 50),
    ]

    for strategy, size, overlap in strategies:
        chunker = Chunker(chunk_size=size, overlap=overlap, strategy=strategy)
        chunks = chunker.chunk(sample_text)

        print(f"\n--- {strategy.value} (size={size}, overlap={overlap}) ---")
        print(f"Total chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            preview = chunk.text[:80].replace('\n', ' ')
            print(f"  [{i}] {chunk.token_count} tokens: {preview}...")
