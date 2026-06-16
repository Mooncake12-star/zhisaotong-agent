import os
import sys
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestMCPServer:
    def test_mcp_fastmcp_import(self):
        from mcp.server.fastmcp import FastMCP
        assert FastMCP is not None

    def test_mcp_stdlib_import(self):
        import mcp.types as types
        assert types is not None


class TestModelFactory:
    def test_factory_config(self):
        from utils.config_handler import rag_conf
        assert rag_conf["chat_model_name"] != ""
        assert rag_conf["embedding_model_name"] != ""
