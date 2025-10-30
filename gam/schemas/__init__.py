"""
Schemas Module

This module exposes all core data models and protocol definitions for the GAM (General-Agentic-Memory) framework.
It organizes memory, page, search, tool, and result schemas for unified import and type safety across the system.
"""
from .memory import MemoryState, MemoryUpdate, MemoryStore, InMemoryMemoryStore
from .page import Page, PageStore, InMemoryPageStore
from .search import SearchPlan, Retriever, Hit
from .tools import ToolResult, Tool, ToolRegistry
from .result import Result, EnoughDecision, ReflectionDecision, ResearchOutput, GenerateRequests

# JSON Schema constants for LLM and system validation
PLANNING_SCHEMA = SearchPlan.model_json_schema()
INTEGRATE_SCHEMA = Result.model_json_schema()
INFO_CHECK_SCHEMA = EnoughDecision.model_json_schema()
GENERATE_REQUESTS_SCHEMA = GenerateRequests.model_json_schema()

__all__ = [
    "MemoryState", "MemoryUpdate", "MemoryStore", "InMemoryMemoryStore",
    "Page", "PageStore", "InMemoryPageStore",
    "SearchPlan", "Retriever", "Hit",
    "ToolResult", "Tool", "ToolRegistry",
    "Result", "EnoughDecision", "ReflectionDecision", "ResearchOutput", "GenerateRequests",
    "PLANNING_SCHEMA", "INTEGRATE_SCHEMA", "INFO_CHECK_SCHEMA", "GENERATE_REQUESTS_SCHEMA",
]
