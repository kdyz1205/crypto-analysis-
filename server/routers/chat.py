"""
Chat routes: AI conversation with Claude.
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ..core.dependencies import get_chat

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: str | None = None


@router.post("")
async def api_chat(req: ChatRequest):
    """Send a message to the AI and get a response."""
    chat = get_chat()
    result = await chat.chat(req.message, req.session_id, req.model)
    return result


@router.get("/models")
async def api_chat_models():
    """List available AI models."""
    return get_chat().list_models()


@router.get("/history")
async def api_chat_history(session_id: str = Query("default")):
    """Get chat history for a session."""
    session = get_chat().get_session(session_id)
    return {"messages": session.messages, "model": session.model}


@router.post("/clear")
async def api_chat_clear(session_id: str = Query("default")):
    """Clear chat history."""
    get_chat().clear_session(session_id)
    return {"ok": True}
