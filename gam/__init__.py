# -*- coding: utf-8 -*-
"""
General Agentic Memory (GAM) Framework

A dual-agent architecture for building long-term memory with deep research capabilities.

Key Components:
- MemoryAgent: Builds structured memory from raw messages
- ResearchAgent: Performs multi-iteration research with reflection
"""

from __future__ import annotations

# Core agents
from gam.agents import MemoryAgent, ResearchAgent

# Generators
from gam.generator import AbsGenerator, OpenAIGenerator

# Retrievers
from gam.retriever import (
    AbsRetriever,
    IndexRetriever
)

# Configurations
from gam.config import (
    OpenAIGeneratorConfig,
    VLLMGeneratorConfig,
    DenseRetrieverConfig,
    BM25RetrieverConfig,
    IndexRetrieverConfig
)

# Schemas
from gam.schemas import (
    MemoryState,
    Page,
    MemoryUpdate,
    SearchPlan,
    Hit,
    Result,
    ReflectionDecision,
    ResearchOutput,
    InMemoryMemoryStore,
    InMemoryPageStore
)

__version__ = "0.1.0"
__all__ = [
    # Core agents
    "MemoryAgent",
    "ResearchAgent",
    
    # Generators
    "AbsGenerator",
    "OpenAIGenerator",
    "VLLMGenerator",
    
    # Retrievers
    "AbsRetriever",
    "IndexRetriever",
    
    # Configurations
    "OpenAIGeneratorConfig",
    "VLLMGeneratorConfig",
    "DenseRetrieverConfig",
    "BM25RetrieverConfig",
    "IndexRetrieverConfig",
    
    # Schemas
    "MemoryState",
    "Page",
    "MemoryUpdate",
    "SearchPlan",
    "Hit",
    "Result",
    "ReflectionDecision",
    "ResearchOutput",
    "InMemoryMemoryStore",
    "InMemoryPageStore",
]

