# =============================================================================
# TESTES - ChatAgent Module
# =============================================================================
# Testes unitários para o ChatAgent e componentes relacionados
# =============================================================================

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestStreamChunk:
    """Testes para StreamChunk dataclass."""

    def test_to_sse_text(self):
        """Verifica SSE com texto."""
        from agents.chat_agent import StreamChunk

        chunk = StreamChunk(text="Hello")
        sse = chunk.to_sse()

        assert sse == 'data: {"text": "Hello"}\n\n'

    def test_to_sse_session_id(self):
        """Verifica SSE com session_id."""
        from agents.chat_agent import StreamChunk

        chunk = StreamChunk(session_id="abc-123")
        sse = chunk.to_sse()

        assert '"session_id": "abc-123"' in sse

    def test_to_sse_error(self):
        """Verifica SSE com erro."""
        from agents.chat_agent import StreamChunk

        chunk = StreamChunk(error="Something went wrong")
        sse = chunk.to_sse()

        assert '"error": "Something went wrong"' in sse

    def test_to_sse_done(self):
        """Verifica SSE de finalização."""
        from agents.chat_agent import StreamChunk

        chunk = StreamChunk(done=True)
        sse = chunk.to_sse()

        assert sse == "data: [DONE]\n\n"

    def test_to_sse_refresh_sessions(self):
        """Verifica SSE com refresh_sessions."""
        from agents.chat_agent import StreamChunk

        chunk = StreamChunk(refresh_sessions=True, command="favorite")
        sse = chunk.to_sse()

        assert '"refresh_sessions": true' in sse
        assert '"command": "favorite"' in sse

    def test_to_sse_combined(self):
        """Verifica SSE com múltiplos campos."""
        from agents.chat_agent import StreamChunk

        chunk = StreamChunk(text="Hi", session_id="123")
        sse = chunk.to_sse()

        assert '"text": "Hi"' in sse
        assert '"session_id": "123"' in sse


class TestChatRequest:
    """Testes para ChatRequest dataclass."""

    def test_default_values(self):
        """Verifica valores padrão."""
        from agents.chat_agent import ChatRequest

        request = ChatRequest(message="hello")

        assert request.message == "hello"
        assert request.session_id is None
        assert request.model == "opus"
        assert request.resume is True
        assert request.fork_session is None
        assert request.project == "default"

    def test_custom_values(self):
        """Verifica valores customizados."""
        from agents.chat_agent import ChatRequest

        request = ChatRequest(
            message="test",
            session_id="abc-123",
            model="haiku",
            resume=False,
            project="my-project",
        )

        assert request.session_id == "abc-123"
        assert request.model == "haiku"
        assert request.resume is False
        assert request.project == "my-project"


class TestDetectSessionCommand:
    """Testes para detecção de comandos de sessão."""

    def test_detect_favorite(self):
        """Verifica detecção de favoritar."""
        from agents.chat_agent import detect_session_command

        assert detect_session_command("favoritar")[0] == "favorite"
        assert detect_session_command("favorite esse chat")[0] == "favorite"
        assert detect_session_command("adiciona aos favoritos")[0] == "favorite"
        assert detect_session_command("marca como favorito")[0] == "favorite"

    def test_detect_unfavorite(self):
        """Verifica detecção de desfavoritar."""
        from agents.chat_agent import detect_session_command

        assert detect_session_command("desfavoritar")[0] == "unfavorite"
        assert detect_session_command("tirar dos favoritos")[0] == "unfavorite"
        assert detect_session_command("remover de favoritos")[0] == "unfavorite"

    def test_detect_rename(self):
        """Verifica detecção de renomear."""
        from agents.chat_agent import detect_session_command

        cmd, name = detect_session_command("renomear para Teste")
        assert cmd == "rename"
        assert name == "teste"

        cmd, name = detect_session_command("mudar nome para 'Meu Chat'")
        assert cmd == "rename"
        assert "meu chat" in name.lower()

    def test_no_command(self):
        """Verifica mensagem sem comando."""
        from agents.chat_agent import detect_session_command

        assert detect_session_command("olá, como vai?") == (None, None)
        assert detect_session_command("me explique sobre Python") == (None, None)

    def test_unfavorite_before_favorite(self):
        """Verifica que desfavoritar tem prioridade sobre favoritar."""
        from agents.chat_agent import detect_session_command

        # "desfavoritar" contém "favoritar", mas deve detectar unfavorite
        cmd, _ = detect_session_command("desfavoritar")
        assert cmd == "unfavorite"


class TestExecuteSessionCommand:
    """Testes para execução de comandos de sessão."""

    @pytest.mark.asyncio
    async def test_favorite_command(self):
        """Verifica comando de favoritar."""
        from agents.chat_agent import execute_session_command

        mock_afs = MagicMock()
        mock_afs.kv.set = AsyncMock()

        result = await execute_session_command(mock_afs, "session-123", "favorite", None)

        assert "favoritos" in result.lower()
        mock_afs.kv.set.assert_called_once_with("session:favorite", True)

    @pytest.mark.asyncio
    async def test_unfavorite_command(self):
        """Verifica comando de desfavoritar."""
        from agents.chat_agent import execute_session_command

        mock_afs = MagicMock()
        mock_afs.kv.set = AsyncMock()

        result = await execute_session_command(mock_afs, "session-123", "unfavorite", None)

        assert "removido" in result.lower()
        mock_afs.kv.set.assert_called_once_with("session:favorite", False)

    @pytest.mark.asyncio
    async def test_rename_command(self):
        """Verifica comando de renomear."""
        from agents.chat_agent import execute_session_command

        mock_afs = MagicMock()
        mock_afs.kv.set = AsyncMock()

        result = await execute_session_command(mock_afs, "session-123", "rename", "Novo Nome")

        assert "Novo Nome" in result
        mock_afs.kv.set.assert_called_once_with("session:title", "Novo Nome")

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        """Verifica comando desconhecido."""
        from agents.chat_agent import execute_session_command

        mock_afs = MagicMock()

        result = await execute_session_command(mock_afs, "session-123", "invalid", None)

        assert "nao reconhecido" in result.lower()


class TestAppendToJsonl:
    """Testes para persistência JSONL."""

    def test_creates_file_if_not_exists(self, tmp_path):
        """Verifica criação de arquivo."""
        from agents.chat_agent import append_to_jsonl
        import json

        # Patch SESSIONS_DIR
        with patch("agents.chat_agent.SESSIONS_DIR", tmp_path):
            append_to_jsonl("test-session", "user msg", "assistant msg")

        jsonl_file = tmp_path / "test-session.jsonl"
        assert jsonl_file.exists()

        # Verificar conteúdo
        lines = jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 2

        user_entry = json.loads(lines[0])
        assert user_entry["type"] == "user"
        assert user_entry["message"]["content"] == "user msg"

        assistant_entry = json.loads(lines[1])
        assert assistant_entry["type"] == "assistant"

    def test_appends_to_existing_file(self, tmp_path):
        """Verifica append em arquivo existente."""
        from agents.chat_agent import append_to_jsonl

        jsonl_file = tmp_path / "existing-session.jsonl"
        jsonl_file.write_text('{"existing": "data"}\n')

        with patch("agents.chat_agent.SESSIONS_DIR", tmp_path):
            append_to_jsonl("existing-session", "new user", "new assistant")

        lines = jsonl_file.read_text().strip().split("\n")
        assert len(lines) == 3  # 1 existente + 2 novas


class TestChatAgent:
    """Testes para a classe ChatAgent."""

    def test_create_chat_agent(self):
        """Verifica factory function."""
        from agents.chat_agent import create_chat_agent

        agent = create_chat_agent()
        assert agent is not None
        assert agent.rag_search_fn is None

    def test_create_chat_agent_with_rag(self):
        """Verifica factory function com RAG."""
        from agents.chat_agent import create_chat_agent

        async def mock_rag(query):
            return "context"

        agent = create_chat_agent(rag_search_fn=mock_rag)
        assert agent.rag_search_fn is mock_rag

    def test_build_query_message_without_rag(self):
        """Verifica construção de mensagem sem RAG."""
        from agents.chat_agent import ChatAgent

        agent = ChatAgent()
        result = agent._build_query_message("hello", None, "/path/to/artifacts")

        assert result == "hello"

    def test_build_query_message_with_rag(self):
        """Verifica construção de mensagem com RAG."""
        from agents.chat_agent import ChatAgent

        agent = ChatAgent()
        result = agent._build_query_message(
            "hello", "some context from RAG", "/path/to/artifacts"
        )

        assert "hello" in result
        assert "some context from RAG" in result
        assert "<base_conhecimento>" in result
        assert "/path/to/artifacts" in result

    def test_build_query_message_ignores_warning(self):
        """Verifica que ignora contexto RAG com aviso."""
        from agents.chat_agent import ChatAgent

        agent = ChatAgent()
        result = agent._build_query_message(
            "hello", "[AVISO] Nenhum resultado encontrado", "/path"
        )

        assert result == "hello"
        assert "<base_conhecimento>" not in result

    @pytest.mark.asyncio
    async def test_stream_validates_session_id(self):
        """Verifica validação de session_id."""
        from agents.chat_agent import ChatAgent, ChatRequest

        agent = ChatAgent()
        request = ChatRequest(message="test", session_id="invalid-session")

        chunks = []
        async for chunk in agent.stream(request):
            chunks.append(chunk)

        # Deve retornar erro de validação
        assert any(c.error for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_with_session_command(self):
        """Verifica processamento de comando de sessão."""
        from agents.chat_agent import ChatAgent, ChatRequest

        # Mocks
        mock_afs = MagicMock()
        mock_afs.kv.set = AsyncMock()
        mock_afs.kv.get = AsyncMock(return_value=[])
        mock_afs.close = AsyncMock()

        mock_client = MagicMock()

        with patch("agents.chat_agent.validate_session_id"), \
             patch("agents.chat_agent.AgentFS") as MockAgentFS, \
             patch("agents.chat_agent.get_client", return_value=mock_client), \
             patch("agents.chat_agent.set_current_session_id"), \
             patch("agents.chat_agent.append_to_jsonl"), \
             patch("agents.chat_agent.reset_session") as mock_reset:

            MockAgentFS.open = AsyncMock(return_value=mock_afs)

            # Simular app_state via import dentro da função
            import app_state
            original_client = app_state.client
            original_session_id = app_state.current_session_id
            app_state.client = mock_client
            app_state.current_session_id = "new-uuid-123"

            try:
                agent = ChatAgent()
                request = ChatRequest(message="favoritar", session_id=None)

                chunks = []
                async for chunk in agent.stream(request):
                    chunks.append(chunk)

                # Deve ter session_id, texto de confirmação e done
                assert any(c.session_id for c in chunks)
                assert any(c.text and "favoritos" in c.text.lower() for c in chunks)
                assert any(c.done for c in chunks)
                assert any(c.refresh_sessions for c in chunks)
            finally:
                # Restaurar estado original
                app_state.client = original_client
                app_state.current_session_id = original_session_id


class TestChatAgentIntegration:
    """Testes de integração do ChatAgent (com mocks de dependências externas)."""

    @pytest.mark.asyncio
    async def test_full_chat_flow_mocked(self):
        """Verifica fluxo completo de chat com mocks."""
        from agents.chat_agent import ChatAgent, ChatRequest
        from dataclasses import dataclass

        # Mocks
        mock_afs = MagicMock()
        mock_afs.kv.set = AsyncMock()
        mock_afs.kv.get = AsyncMock(return_value=[])
        mock_afs.tools.start = AsyncMock(return_value="call-123")
        mock_afs.tools.finish = AsyncMock()
        mock_afs.close = AsyncMock()

        # Mock do client do Claude
        mock_client = MagicMock()
        mock_client.query = AsyncMock()

        # Mock da resposta do Claude
        @dataclass
        class MockTextBlock:
            text: str = "Ola! Como posso ajudar?"

        @dataclass
        class MockMessage:
            content: list = None

            def __post_init__(self):
                self.content = [MockTextBlock()]

        async def mock_receive():
            yield MockMessage()

        mock_client.receive_response = mock_receive

        with patch("agents.chat_agent.validate_session_id"), \
             patch("agents.chat_agent.AgentFS") as MockAgentFS, \
             patch("agents.chat_agent.get_client", return_value=mock_client), \
             patch("agents.chat_agent.set_current_session_id"), \
             patch("agents.chat_agent.append_to_jsonl"), \
             patch("agents.chat_agent.reset_session"):

            MockAgentFS.open = AsyncMock(return_value=mock_afs)

            # Mock AssistantMessage e TextBlock para isinstance check
            with patch("claude_agent_sdk.AssistantMessage", MockMessage), \
                 patch("claude_agent_sdk.TextBlock", MockTextBlock):

                # Simular app_state
                import app_state
                original_client = app_state.client
                original_session_id = app_state.current_session_id
                app_state.client = mock_client
                app_state.current_session_id = "test-uuid-456"

                try:
                    agent = ChatAgent()
                    request = ChatRequest(message="ola", session_id=None)

                    chunks = []
                    async for chunk in agent.stream(request):
                        chunks.append(chunk)

                    # Verificações
                    assert len(chunks) > 0
                    assert any(c.session_id == "test-uuid-456" for c in chunks)
                    assert any(c.done for c in chunks)
                finally:
                    app_state.client = original_client
                    app_state.current_session_id = original_session_id
