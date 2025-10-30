# memory_agent.py
# -*- coding: utf-8 -*-
"""
MemoryAgent Module

This module defines the MemoryAgent for the GAM (General-Agentic-Memory) framework.

- Memory is represented as a list[str] of abstracts (no events/tags included).
- MemoryAgent exposes only: memorize(message) -> MemoryUpdate, allowing the agent to store new information.
- Prompts within the module are used as placeholders for future prompt templates or instructions.
"""


from __future__ import annotations

from typing import Optional, Tuple

from gam.prompts import MemoryAgent_PROMPT
from gam.schemas import (
    MemoryState, Page, MemoryUpdate, MemoryStore, PageStore,
    InMemoryMemoryStore, InMemoryPageStore,
)
from gam.generator import AbsGenerator

class MemoryAgent:
    """
    Public API:
      - memorize(message) -> MemoryUpdate
    Internal only:
      - _decorate(message, memory_state) -> (abstract, header, decorated_new_page)
    Note: memory_state contains ONLY abstracts (list[str]).
    """

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        page_store: PageStore | None = None,
        generator: AbsGenerator | None = None,  # 必须传入Generator实例
        dir_path: Optional[str] = None,  # 新增：文件系统存储路径
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for MemoryAgent")
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.page_store = page_store or InMemoryPageStore(dir_path=dir_path)
        self.generator = generator

    # ---- Public ----
    def memorize(self, message: str) -> MemoryUpdate:
        """
        Update long-term memory with a new message and persist a decorated page.
        Steps:
          1) _decorate(...) => abstract, header, decorated_new_page
          2) Merge into MemoryState (append unique abstract)
          3) Write Page into page_store  (page_id left None by default)
        """
        message = message.strip()
        state = self.memory_store.load()

        # (1) Decorate - this generates the abstract and decorated page
        abstract, header, decorated_new_page = self._decorate(message, state)

        # (2) Add abstract to memory (with built-in uniqueness check)
        self.memory_store.add(abstract)

        # (3) Persist page
        page = Page(header=header, content=message, meta={"decorated": decorated_new_page})
        self.page_store.add(page)
        
        # (4) Get updated state after adding abstract
        updated_state = self.memory_store.load()

        return MemoryUpdate(new_state=updated_state, new_page=page, debug={"decorated_page": decorated_new_page})

    # ---- Internal helpers ----
    def _decorate(self, message: str, memory_state: MemoryState) -> Tuple[str, str, str]:
        """
        Private. Generate abstract for the message and compose: "abstract; header; new_page".
        Returns: (abstract, header, decorated_new_page)
        """
        # Build memory context from existing abstracts
        memory_context = ""
        if memory_state.abstracts:
            memory_context = "\n".join([f"- {abstract}" for abstract in memory_state.abstracts])
        
        # Generate abstract for the current message using LLM with memory context
        prompt = MemoryAgent_PROMPT.format(
            input_message=message,
            memory_context=memory_context
        )
        
        try:
            response = self.generator.generate_single(prompt=prompt)
            abstract = response.get("text", "").strip()
        except Exception as e:
            print(f"Error generating abstract: {e}")
            abstract = message[:200]
        
        # Create header with the new abstract
        header = f"[ABSTRACT] {abstract}".strip()
        decorated_new_page = f"{header}; {message}"
        return abstract, header, decorated_new_page