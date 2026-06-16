from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    query: str
    image_path: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str


class StatusResponse(BaseModel):
    status: str
    mcp_connected: bool
    rag_ready: bool
