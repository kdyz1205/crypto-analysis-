"""
Shared singleton accessors for all routers.

Singletons are initialized once by app.py lifespan event.
Routers import getters from here — never instantiate directly.
"""

from ..agent_brain import AgentBrain
from ..ai_chat import AIChatEngine
from ..self_healer import SelfHealer

_agent: AgentBrain | None = None
_chat: AIChatEngine | None = None
_healer: SelfHealer | None = None


def init_agent() -> AgentBrain:
    global _agent
    if _agent is None:
        _agent = AgentBrain()
    return _agent


def init_chat() -> AIChatEngine:
    global _chat
    if _chat is None:
        _chat = AIChatEngine()
    return _chat


def init_healer() -> SelfHealer:
    global _healer
    if _healer is None:
        _healer = SelfHealer()
    return _healer


def get_agent() -> AgentBrain:
    return init_agent()


def get_chat() -> AIChatEngine:
    return init_chat()


def get_healer() -> SelfHealer:
    return init_healer()
