import os
import sys
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestFastAPISchemas:
    def test_schemas_import(self):
        from fastapi_app.schemas import ChatRequest, ChatResponse, StatusResponse
        assert ChatRequest is not None
        assert ChatResponse is not None
        assert StatusResponse is not None

    def test_chat_request(self):
        from fastapi_app.schemas import ChatRequest
        req = ChatRequest(query="你好")
        assert req.query == "你好"
        assert req.image_path is None

    def test_chat_request_with_image(self):
        from fastapi_app.schemas import ChatRequest
        req = ChatRequest(query="分析图片", image_path="/tmp/test.jpg")
        assert req.image_path == "/tmp/test.jpg"

    def test_chat_response(self):
        from fastapi_app.schemas import ChatResponse
        resp = ChatResponse(answer="这是回答")
        assert resp.answer == "这是回答"

    def test_status_response(self):
        from fastapi_app.schemas import StatusResponse
        resp = StatusResponse(status="ok", mcp_connected=True, rag_ready=False)
        assert resp.status == "ok"
        assert resp.mcp_connected is True
        assert resp.rag_ready is False


class TestAgentManager:
    def test_manager_import(self):
        from fastapi_app.deps import AgentManager
        manager = AgentManager()
        assert manager.is_ready is False
        assert manager.agent is None


class TestReactAgent:
    def test_react_mcp_context_manager(self):
        from agent.react_agent import ReactMCPContextManager
        mgr = ReactMCPContextManager("python", ["test.py"], cwd="/tmp")
        assert mgr.command == "python"
        assert mgr.session is None

    def test_agent_init(self):
        from agent.react_agent import ReactAgent
        agent = ReactAgent()
        assert agent.system_prompt is not None
        assert agent.conversation_history == []
