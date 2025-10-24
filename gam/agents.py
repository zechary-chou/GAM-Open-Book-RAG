
# agents.py
# -*- coding: utf-8 -*-
"""
- Memory == list[str] of abstracts (no events/tags).
- MemoryAgent exposes only: memorize(message) -> MemoryUpdate
- ResearchAgent uses explicit research.
Prompts are placeholders.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import re
import json

from prompts import MemoryAgent_PROMPT, Planning_PROMPT, Integrate_PROMPT, InfoCheck_PROMPT, GenerateRequests_PROMPT
from json_schema import PLANNING_SCHEMA, INTEGRATE_SCHEMA, INFO_CHECK_SCHEMA, GENERATE_REQUESTS_SCHEMA

# =============================
# Core data models
# =============================

@dataclass
class MemoryState:
    """Long-term memory: only abstracts list."""
    abstracts: List[str] = field(default_factory=list)


@dataclass
class Page:
    header: str
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryUpdate:
    new_state: MemoryState
    new_page: Page
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchPlan:
    info_needs: List[str] = field(default_factory=list)      
    tools: List[str] = field(default_factory=list)             
    keyword_collection: List[str] = field(default_factory=list)  
    vector_queries: List[str] = field(default_factory=list)     
    page_indices: List[int] = field(default_factory=list)       


@dataclass
class ToolResult:
    tool: str
    inputs: Dict[str, Any]
    outputs: Any
    error: Optional[str] = None   # None means success; otherwise error info.


@dataclass
class Hit:
    page_index: Optional[int]       # Index in page_store, None for tool results
    snippet: str
    source: str                     # "keyword" | "vector" | "page_index" | "tool:<name>"
    meta: Dict[str, Any] = field(default_factory=dict)




@dataclass
class Result:
    """
    Temporary memory containing question-relevant information.
    This is the result of LLM processing search results into relevant memory.
    """
    content: str = ""  # The integrated memory content about the question
    sources: List[Dict[str, Any]] = field(default_factory=list)  # Source references


@dataclass
class ReflectionDecision:
    enough: bool
    new_request: Optional[str]


@dataclass
class ResearchOutput:
    integrated_memory: str
    raw_memory: Dict[str, Any]


# =============================
# Minimal interfaces / protocols
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
    def build(self, pages: List[Page]) -> None: ...
    def search(self, query: str, top_k: int = 10) -> List[Hit]: ...


class Tool(Protocol):
    name: str
    def run(self, **kwargs) -> ToolResult: ...


class ToolRegistry(Protocol):
    def run_many(self, tool_inputs: Dict[str, Dict[str, Any]]) -> List[ToolResult]: ...


# =============================
# In-memory default stores (for quick start)
# =============================

class InMemoryMemoryStore:
    def __init__(self, init_state: Optional[MemoryState] = None) -> None:
        self._state = init_state or MemoryState()

    def load(self) -> MemoryState:
        return self._state

    def save(self, state: MemoryState) -> None:
        self._state = state

    def add(self, abstract: str) -> None:
        """Add a new abstract to memory if it doesn't already exist."""
        if abstract and abstract not in self._state.abstracts:
            self._state.abstracts.append(abstract)


class InMemoryPageStore:
    """
    Simple append-only list store for Page.
    Uses index-based access.
    """
    def __init__(self) -> None:
        self._pages: List[Page] = []

    def add(self, page: Page) -> None:
        self._pages.append(page)

    def get(self, index: int) -> Optional[Page]:
        if 0 <= index < len(self._pages):
            return self._pages[index]
        return None

    def list_all(self) -> List[Page]:
        return list(self._pages)


# =============================
# MemoryAgent
# =============================

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
        llm: Any = None,  # 必须传入LLM实例
    ) -> None:
        if llm is None:
            raise ValueError("LLM instance is required for MemoryAgent")
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.page_store = page_store or InMemoryPageStore()
        self.llm = llm

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
        self.memory_store.save(state)

        return MemoryUpdate(new_state=state, new_page=page, debug={"decorated_page": decorated_new_page})

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
            response = self.llm.generate(prompt=prompt, max_tokens=512)
            abstract = response.get("text", "").strip()
        except Exception as e:
            print(f"Error generating abstract: {e}")
            abstract = message[:200]
        
        # Create header with the new abstract
        header = f"[ABSTRACT] {abstract}".strip()
        decorated_new_page = f"{header}; {message}"
        return abstract, header, decorated_new_page
    

# =============================
# ResearchAgent
# =============================

class ResearchAgent:
    """
    Public API:
      - research(request) -> ResearchOutput
    Internal steps:
      - _planning(request, memory_state) -> SearchPlan
      - _search(plan) -> SearchResults  (calls keyword/vector/page_id + tools)
      - _integrate(search_results, temp_memory) -> TempMemory
      - _reflection(request, memory_state, temp_memory) -> ReflectionDecision

    Note: Uses MemoryStore to dynamically load current memory state.
    This allows ResearchAgent to access the latest memory updates from MemoryAgent.
    """

    def __init__(
        self,
        page_store: PageStore,
        memory_store: MemoryStore | None = None,
        tool_registry: Optional[ToolRegistry] = None,
        retrievers: Optional[Dict[str, Retriever]] = None,
        llm: Any = None,  # 必须传入LLM实例
        max_iters: int = 3,
    ) -> None:
        if llm is None:
            raise ValueError("LLM instance is required for ResearchAgent")
        self.page_store = page_store
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.tools = tool_registry
        self.retrievers = retrievers or {}
        self.llm = llm
        self.max_iters = max_iters

        # Build indices upfront (if retrievers are provided)
        pages = self.page_store.list_all()
        for r in self.retrievers.values():
            try:
                r.build(pages)
            except Exception:
                pass

    # ---- Public ----
    def research(self, request: str) -> ResearchOutput:
        temp = Result()
        iterations: List[Dict[str, Any]] = []
        next_request = request

        for step in range(self.max_iters):
            # Load current memory state dynamically
            memory_state = self.memory_store.load()
            plan = self._planning(next_request, memory_state)

            temp = self._search(plan, temp, request)

            decision = self._reflection(request, temp)

            iterations.append({
                "step": step,
                "plan": plan.__dict__,
                "temp_memory": temp.__dict__,
                "decision": decision.__dict__,
            })

            if decision.enough:
                break

            if not decision.new_request:
                next_request = request
            else:
                next_request = decision.new_request


        raw = {
            "iterations": iterations,
            "temp_memory": temp.__dict__,
        }
        return ResearchOutput(integrated_memory=temp.content, raw_memory=raw)

    # ---- Internal ----
    def _planning(self, request: str, memory_state: MemoryState) -> SearchPlan:
        """
        Produce a SearchPlan:
          - what specific info is needed
          - which tools are useful + inputs
          - keyword/vector/page_id payloads
        """
        # Build Context - Use memory_state abstracts with page numbering
        if memory_state.abstracts:
            memory_context_lines = []
            for i, abstract in enumerate(memory_state.abstracts):
                memory_context_lines.append(f"Page {i}: {abstract}")
            memory_context = "\n".join(memory_context_lines)
        else:
            memory_context = "No memory currently."
        
        prompt = Planning_PROMPT.format(request=request, memory=memory_context)

        try:
            response = self.llm.generate(prompt=prompt, max_tokens=500, schema=PLANNING_SCHEMA)
            data = response.get("json") or json.loads(response["text"])
            return SearchPlan(
                info_needs=data.get("info_needs", []),
                tools=data.get("tools", []),
                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                page_indices=data.get("page_indices", [])
            )
        except Exception as e:
            print(f"Error in planning: {e}")
            return SearchPlan(
                info_needs=[],
                tools=[],
                keyword_collection=[],
                vector_queries=[],
                page_indices=[]
            )
    

    def _search(self, plan: SearchPlan, temp_memory: Result, question: str) -> Result:
        """
        Unified search with integration:
          1) Execute search tools
          2) Integrate results with LLM
        Returns Result directly.
        """
        hits: List[Hit] = []

        # Execute each planned tool
        for tool in plan.tools:
            if tool == "keyword":
                for query in plan.keyword_collection:
                    hits.extend(self._search_by_keyword(query, top_k=5))
                    
            elif tool == "vector":
                for query in plan.vector_queries:
                    hits.extend(self._search_by_vector(query, top_k=5))
                    
            elif tool == "page_index":
                if plan.page_indices:
                    hits.extend(self._search_by_page_index(plan.page_indices))

        # Integrate search results with LLM
        return self._integrate(hits, temp_memory, question)

    def _integrate(self, hits: List[Hit], temp_memory: Result, question: str) -> Result:
        """
        Integrate search hits with LLM to generate question-relevant result.
        """
        # Build evidence context from search hits
        evidence_text = []
        sources = []
        for i, hit in enumerate(hits, 1):
            evidence_text.append(f"{i}. [{hit.source}] {hit.snippet}")
            sources.append({
                "page_index": hit.page_index,
                "snippet": hit.snippet,
                "source": hit.source
            })
        
        evidence_context = "\n".join(evidence_text) if evidence_text else "无搜索结果"
        
        prompt = Integrate_PROMPT.format(question=question, evidence_context=evidence_context, temp_memory=temp_memory.content)

        try:
            response = self.llm.generate(prompt=prompt, max_tokens=800, schema=INTEGRATE_SCHEMA)
            data = response.get("json") or json.loads(response["text"])
            
            return Result(
                content=data.get("content", ""),
                sources=data.get("sources", sources)
            )
        except Exception as e:
            print(f"Error in integration: {e}")
            return temp_memory

    # ---- search channels ----
    def _search_by_keyword(self, query: str, top_k: int = 10) -> List[Hit]:
        r = self.retrievers.get("keyword")
        if r is not None:
            try:
                return r.search(query, top_k=top_k)
            except Exception:
                return []
        # naive fallback: scan pages for substring
        out: List[Hit] = []
        q = query.lower()
        for i, p in enumerate(self.page_store.list_all()):
            if q in p.content.lower() or q in p.header.lower():
                snippet = p.content[:200]
                out.append(Hit(page_index=i, snippet=snippet, source="keyword", meta={}))
                if len(out) >= top_k:
                    break
        return out

    def _search_by_vector(self, query: str, top_k: int = 10) -> List[Hit]:
        r = self.retrievers.get("vector")
        if r is not None:
            try:
                return r.search(query, top_k=top_k)
            except Exception:
                return []
        # fallback: none
        return []

    def _search_by_page_index(self, page_indices: List[int]) -> List[Hit]:
        out: List[Hit] = []
        for idx in page_indices:
            p = self.page_store.get(idx)
            if p:
                out.append(Hit(page_index=idx, snippet=p.content[:200], source="page_index", meta={}))
        return out
        

    # ---- reflection & summarization ----
    def _reflection(self, request: str, temp_memory: Result) -> ReflectionDecision:
        """
        - "whether information is enough" 
        - "if not, generate remaining information as a new request"  
        """
        
        try:
            # Step 1: Check for completeness of information
            check_prompt = InfoCheck_PROMPT.format(request=request, temp_memory=temp_memory.content)
            check_response = self.llm.generate(prompt=check_prompt, max_tokens=256, schema=INFO_CHECK_SCHEMA)
            check_data = check_response.get("json") or json.loads(check_response["text"])
            
            enough = check_data.get("enough", False)
            
            # If there is enough information, return directly
            if enough:
                return ReflectionDecision(enough=True, new_request=None)
            
            # Step 2: Generate a list of new requests
            generate_prompt = GenerateRequests_PROMPT.format(
                request=request, 
                temp_memory=temp_memory.content
            )
            generate_response = self.llm.generate(prompt=generate_prompt, max_tokens=512, schema=GENERATE_REQUESTS_SCHEMA)
            generate_data = generate_response.get("json") or json.loads(generate_response["text"])
            
            # Splices the list of requests into a string
            new_requests_list = generate_data.get("new_requests", [])
            new_request = None
            
            if new_requests_list and isinstance(new_requests_list, list):
                new_request = " ".join(new_requests_list)
            
            return ReflectionDecision(
                enough=False,
                new_request=new_request
            )
            
        except Exception as e:
            print(f"Error in reflection: {e}")
            return ReflectionDecision(enough=False, new_request=None)