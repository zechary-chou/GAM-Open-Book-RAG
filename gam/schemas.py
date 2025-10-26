# schemas.py
# -*- coding: utf-8 -*-
"""
统一的数据模型定义
使用 Pydantic 实现一个定义，多种用途：
- Python 数据类
- JSON Schema 自动生成
- 数据验证
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Protocol

# =============================
# Core data models (Pydantic)
# =============================

class MemoryState(BaseModel):
    """Long-term memory: only abstracts list."""
    abstracts: List[str] = Field(default_factory=list, description="List of memory abstracts")

class Page(BaseModel):
    """Page data structure"""
    header: str = Field(..., description="Page header")
    content: str = Field(..., description="Page content")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    
    @staticmethod
    def equal(page1: 'Page', page2: 'Page') -> bool:
        """判断两个 Page 是否相等"""
        return page1 == page2

class MemoryUpdate(BaseModel):
    """Memory update result"""
    new_state: MemoryState = Field(..., description="Updated memory state")
    new_page: Page = Field(..., description="New page added")
    debug: Dict[str, Any] = Field(default_factory=dict, description="Debug information")

class SearchPlan(BaseModel):
    """Search planning structure"""
    info_needs: List[str] = Field(default_factory=list, description="List of information needs")
    tools: List[str] = Field(default_factory=list, description="Tools to use for searching")
    keyword_collection: List[str] = Field(default_factory=list, description="Keywords to search for")
    vector_queries: List[str] = Field(default_factory=list, description="Semantic search queries")
    page_index: List[int] = Field(default_factory=list, description="Specific page indices to retrieve")

class ToolResult(BaseModel):
    """Tool execution result"""
    tool: str = Field(..., description="Tool name")
    inputs: Dict[str, Any] = Field(..., description="Input parameters")
    outputs: Any = Field(..., description="Output results")
    error: Optional[str] = Field(None, description="Error message if any")

class Hit(BaseModel):
    """Search result hit"""
    page_id: Optional[str] = Field(None, description="Page ID in store")
    snippet: str = Field(..., description="Text snippet from the source")
    source: str = Field(..., description="Source type (keyword/vector/page_index/tool)")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class SourceInfo(BaseModel):
    """Source information"""
    page_id: Optional[str] = Field(None, description="Page ID in store")
    snippet: str = Field(..., description="Text snippet from the source")
    source: str = Field(..., description="Source type")

class Result(BaseModel):
    """Search and integration result"""
    content: str = Field("", description="Integrated content about the question")
    sources: List[SourceInfo] = Field(default_factory=list, description="List of sources used")

class ReflectionDecision(BaseModel):
    """Reflection decision"""
    enough: bool = Field(..., description="Whether information is sufficient")
    new_request: Optional[str] = Field(None, description="New request if information is insufficient")

class ResearchOutput(BaseModel):
    """Research output"""
    integrated_memory: str = Field(..., description="Integrated memory content")
    raw_memory: Dict[str, Any] = Field(..., description="Raw memory data")

class GenerateRequests(BaseModel):
    """Generate new requests"""
    new_requests: List[str] = Field(..., description="List of new search requests")

# =============================
# Protocols (Interface definitions)
# =============================

class MemoryStore(Protocol):
    def load(self) -> MemoryState: ...
    def save(self, state: MemoryState) -> None: ...
    def add(self, abstract: str) -> None: ...

class PageStore(Protocol):
    def add(self, page: Page) -> None: ...
    def get(self, index: int) -> Optional[Page]: ...
    def list_all(self) -> List[Page]: ...

class Retriever(Protocol):
    """Unified interface for keyword / vector / page-id retrievers."""
    name: str
    def build(self, page_store) -> None: ...
    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]: ...

class Tool(Protocol):
    name: str
    def run(self, **kwargs) -> ToolResult: ...

class ToolRegistry(Protocol):
    def run_many(self, tool_inputs: Dict[str, Dict[str, Any]]) -> List[ToolResult]: ...

# =============================
# In-memory default stores (for quick start)
# =============================

class InMemoryMemoryStore:
    def __init__(self, dir_path: Optional[str] = None, init_state: Optional[MemoryState] = None) -> None:
        self._dir_path = Path(dir_path) if dir_path else None
        self._state = init_state or MemoryState()
        if self._dir_path:
            self._memory_file = self._dir_path / "memory_state.json"
            # 如果目录存在，尝试加载现有状态
            if self._memory_file.exists():
                self._state = self.load()

    def load(self) -> MemoryState:
        if self._dir_path and self._memory_file.exists():
            try:
                with open(self._memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return MemoryState(**data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Failed to load memory state from {self._memory_file}: {e}")
                return MemoryState()
        return self._state

    def save(self, state: MemoryState) -> None:
        self._state = state
        if self._dir_path:
            # 确保目录存在
            self._dir_path.mkdir(parents=True, exist_ok=True)
            try:
                with open(self._memory_file, 'w', encoding='utf-8') as f:
                    json.dump(state.model_dump(), f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save memory state to {self._memory_file}: {e}")

    def add(self, abstract: str) -> None:
        """Add a new abstract to memory if it doesn't already exist."""
        if abstract and abstract not in self._state.abstracts:
            self._state.abstracts.append(abstract)
            # 自动保存到文件
            if self._dir_path:
                self.save(self._state)

class InMemoryPageStore:
    """
    Simple append-only list store for Page.
    Uses index-based access with file system persistence.
    """
    def __init__(self, dir_path: Optional[str] = None) -> None:
        self._dir_path = Path(dir_path) if dir_path else None
        self._pages: List[Page] = []
        if self._dir_path:
            self._pages_file = self._dir_path / "pages.json"
            # 如果文件存在，尝试加载现有页面
            if self._pages_file.exists():
                self._load_pages()

    def _load_pages(self) -> None:
        """从文件系统加载所有页面"""
        if not self._dir_path or not self._pages_file.exists():
            return
        
        try:
            with open(self._pages_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 假设存储的是页面列表
                if isinstance(data, list):
                    self._pages = [Page(**page_data) for page_data in data]
                else:
                    # 如果存储的是字典格式，尝试获取pages字段
                    self._pages = [Page(**page_data) for page_data in data.get('pages', [])]
        except Exception as e:
            print(f"Warning: Failed to load pages from {self._pages_file}: {e}")

    def _save_pages(self) -> None:
        """保存所有页面到文件"""
        if not self._dir_path:
            return
        
        # 确保目录存在
        self._dir_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 将所有页面转换为字典列表
            pages_data = [page.model_dump() for page in self._pages]
            with open(self._pages_file, 'w', encoding='utf-8') as f:
                json.dump(pages_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save pages to {self._pages_file}: {e}")

    def add(self, page: Page) -> None:
        self._pages.append(page)
        # 保存到文件系统
        if self._dir_path:
            self._save_pages()

    def get(self, index: int) -> Optional[Page]:
        if 0 <= index < len(self._pages):
            return self._pages[index]
        return None

    def list_all(self) -> List[Page]:
        return list(self._pages)
    
    def save(self, dir_path: str) -> None:
        """保存页面到指定目录"""
        save_path = Path(dir_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            pages_data = [page.model_dump() for page in self._pages]
            with open(save_path / "pages.json", 'w', encoding='utf-8') as f:
                json.dump(pages_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save pages to {save_path}: {e}")
    
    @classmethod
    def load(cls, dir_path: str) -> "InMemoryPageStore":
        """从指定目录加载页面"""
        store = cls(dir_path=dir_path)
        return store

# =============================
# Auto-generated JSON Schema
# =============================

# JSON Schema for LLM calls
PLANNING_SCHEMA = SearchPlan.model_json_schema()
INTEGRATE_SCHEMA = Result.model_json_schema()
INFO_CHECK_SCHEMA = ReflectionDecision.model_json_schema()
GENERATE_REQUESTS_SCHEMA = GenerateRequests.model_json_schema()