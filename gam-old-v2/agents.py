
# agents.py
# -*- coding: utf-8 -*-
"""
- Memory == list[str] of abstracts (no events/tags).
- MemoryAgent exposes only: memorize(message) -> MemoryUpdate
- ResearchAgent uses explicit research.
Prompts are placeholders.
"""


from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json

from gam.prompts import MemoryAgent_PROMPT, Planning_PROMPT, Integrate_PROMPT, InfoCheck_PROMPT, GenerateRequests_PROMPT
from gam.schemas import (
    MemoryState, Page, MemoryUpdate, SearchPlan, Hit, Result, 
    EnoughDecision, ReflectionDecision, ResearchOutput, MemoryStore, PageStore, Retriever, 
    ToolRegistry, InMemoryMemoryStore, InMemoryPageStore,
    PLANNING_SCHEMA, INTEGRATE_SCHEMA, INFO_CHECK_SCHEMA, GENERATE_REQUESTS_SCHEMA
)
from gam.generator import AbsGenerator

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
        generator: AbsGenerator | None = None,  # 必须传入Generator实例
        max_iters: int = 3,
        dir_path: Optional[str] = None,  # 新增：文件系统存储路径
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for ResearchAgent")
        self.page_store = page_store
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.tools = tool_registry
        self.retrievers = retrievers or {}
        self.generator = generator
        self.max_iters = max_iters

        # Build indices upfront (if retrievers are provided)
        for name, r in self.retrievers.items():
            try:
                # 调用 retriever 的 build 方法，传递 page_store
                r.build(self.page_store)
                print(f"Successfully built {name} retriever")
            except Exception as e:
                print(f"Failed to build {name} retriever: {e}")
                pass

    # ---- Public ----
    def research(self, request: str) -> ResearchOutput:
        # 在开始研究前，确保检索器索引是最新的
        self._update_retrievers()
        
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

    def _update_retrievers(self):
        """确保检索器索引是最新的"""
        # 检查是否有新的页面需要更新索引
        current_page_count = len(self.page_store.load())
        
        # 如果页面数量发生变化，更新所有检索器索引
        if hasattr(self, '_last_page_count') and current_page_count != self._last_page_count:
            print(f"检测到页面数量变化 ({self._last_page_count} -> {current_page_count})，更新检索器索引...")
            for name, retriever in self.retrievers.items():
                try:
                    retriever.update(self.page_store)
                    print(f"✅ Updated {name} retriever index")
                except Exception as e:
                    print(f"❌ Failed to update {name} retriever: {e}")
        
        # 更新页面计数
        self._last_page_count = current_page_count

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
            response = self.generator.generate_single(prompt=prompt, schema=PLANNING_SCHEMA)
            data = response.get("json") or json.loads(response["text"])
            return SearchPlan(
                info_needs=data.get("info_needs", []),
                tools=data.get("tools", []),
                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                page_index=data.get("page_index", [])
            )
        except Exception as e:
            print(f"Error in planning: {e}")
            return SearchPlan(
                info_needs=[],
                tools=[],
                keyword_collection=[],
                vector_queries=[],
                page_index=[]
            )
    

    def _search(self, plan: SearchPlan, result: Result, question: str) -> Result:
        """
        Unified search with integration:
          1) Execute search tools
          2) Integrate results with LLM
        Returns Result directly.
        """
        hits: List[Hit] = []
        # 用于去重：追踪已添加的 page_id
        added_page_ids = set()

        # Execute each planned tool
        for tool in plan.tools:
            if tool == "keyword":
                for query in plan.keyword_collection:
                    keyword_results = self._search_by_keyword([query], top_k=5)
                    # Flatten the results if they come as List[List[Hit]]
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            for hit in result_list:
                                # 使用 page_id 作为唯一标识
                                page_id = hit.page_id or "unknown"
                                if page_id not in added_page_ids:
                                    hits.append(hit)
                                    added_page_ids.add(page_id)
                    else:
                        for hit in keyword_results:
                            page_id = hit.page_id or "unknown"
                            if page_id not in added_page_ids:
                                hits.append(hit)
                                added_page_ids.add(page_id)
                    
            elif tool == "vector":
                for query in plan.vector_queries:
                    vector_results = self._search_by_vector([query], top_k=5)
                    # Flatten the results if they come as List[List[Hit]]
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            for hit in result_list:
                                page_id = hit.page_id or "unknown"
                                if page_id not in added_page_ids:
                                    hits.append(hit)
                                    added_page_ids.add(page_id)
                    else:
                        for hit in vector_results:
                            page_id = hit.page_id or "unknown"
                            if page_id not in added_page_ids:
                                hits.append(hit)
                                added_page_ids.add(page_id)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index)
                    # Flatten the results if they come as List[List[Hit]]
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            for hit in result_list:
                                page_id = hit.page_id or "unknown"
                                if page_id not in added_page_ids:
                                    hits.append(hit)
                                    added_page_ids.add(page_id)
                    else:
                        for hit in page_results:
                            page_id = hit.page_id or "unknown"
                            if page_id not in added_page_ids:
                                hits.append(hit)
                                added_page_ids.add(page_id)

        # Integrate search results with LLM
        return self._integrate(hits, result, question)

    def _integrate(self, hits: List[Hit], result: Result, question: str) -> Result:
        """
        Integrate search hits with LLM to generate question-relevant result.
        """
        # Build evidence context from search hits
        evidence_text = []
        sources = []
        for i, hit in enumerate(hits, 1):
            # Include page_id in evidence text if available
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            evidence_text.append(f"{i}. {source_info} {hit.snippet}")
            
            if hit.page_id:
                sources.append(hit.page_id)
        
        evidence_context = "\n".join(evidence_text) if evidence_text else "无搜索结果"
        
        prompt = Integrate_PROMPT.format(question=question, evidence_context=evidence_context, result=result.content)

        try:
            response = self.generator.generate_single(prompt=prompt, schema=INTEGRATE_SCHEMA)
            data = response.get("json") or json.loads(response["text"])
            
            return Result(
                content=data.get("content", ""),
                sources=data.get("sources", sources)
            )
        except Exception as e:
            print(f"Error in integration: {e}")
            return result

    # ---- search channels ----
    def _search_by_keyword(self, query_list: List[str], top_k: int = 5) -> List[List[Hit]]:
        r = self.retrievers.get("keyword")
        if r is not None:
            try:
                # BM25Retriever 返回 List[List[Hit]]
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in keyword search: {e}")
                return []
        # naive fallback: scan pages for substring
        out: List[List[Hit]] = []
        for query in query_list:
            query_hits: List[Hit] = []
            q = query.lower()
            for i, p in enumerate(self.page_store.load()):
                if q in p.content.lower() or q in p.header.lower():
                    snippet = p.content
                    query_hits.append(Hit(page_id=str(i), snippet=snippet, source="keyword", meta={}))
                    if len(query_hits) >= top_k:
                        break
            out.append(query_hits)
        return out

    def _search_by_vector(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        r = self.retrievers.get("vector")
        if r is not None:
            try:
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in vector search: {e}")
                return []
        # fallback: none
        return []

    def _search_by_page_index(self, page_index: List[int]) -> List[List[Hit]]:
        r = self.retrievers.get("page_index")
        if r is not None:
            try:
                # IndexRetriever 现在期望 List[str]，将 page_index 转换为逗号分隔的字符串
                query_string = ",".join([str(idx) for idx in page_index])
                hits = r.search([query_string], top_k=len(page_index))
                return hits if hits else []
            except Exception as e:
                print(f"Error in page index search: {e}")
                return []
        
        # fallback: 直接通过 page_store 获取页面
        out: List[Hit] = []
        for idx in page_index:
            p = self.page_store.get(idx)
            if p:
                out.append(Hit(page_id=str(idx), snippet=p.content, source="page_index", meta={}))
        return [out]  # 包装成 List[List[Hit]] 格式
        

    # ---- reflection & summarization ----
    def _reflection(self, request: str, result: Result) -> ReflectionDecision:
        """
        - "whether information is enough" 
        - "if not, generate remaining information as a new request"  
        """
        
        try:
            # Step 1: Check for completeness of information
            check_prompt = InfoCheck_PROMPT.format(request=request, result=result.content)
            check_response = self.generator.generate_single(prompt=check_prompt, schema=INFO_CHECK_SCHEMA)
            check_data = check_response.get("json") or json.loads(check_response["text"])
            
            enough = check_data.get("enough", False)
            
            # If there is enough information, return directly
            if enough:
                return ReflectionDecision(enough=True, new_request=None)
            
            # Step 2: Generate a list of new requests
            generate_prompt = GenerateRequests_PROMPT.format(
                request=request, 
                result=result.content
            )
            generate_response = self.generator.generate_single(prompt=generate_prompt, schema=GENERATE_REQUESTS_SCHEMA)
            generate_data = generate_response.get("json") or json.loads(generate_response["text"])
            
            # Get the list of requests and convert to string
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