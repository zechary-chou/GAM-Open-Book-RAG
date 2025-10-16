import json
import re
import math
import collections

from typing import List, Dict, Tuple, Iterable, Optional

from tqdm import tqdm

import unicodedata
from collections import Counter, defaultdict

from llm_call import *  # expects an object with `.generate(prompt: str) -> str`
from prompts import (
    MemoryAgent_PROMPT, 
    SESSION_SUMMARY_PROMPT,
    PLANNING_DEEP_RESEARCH_PROMPT,
    REPLAN_FROM_SESSIONS_PROMPT
)



# =============== 工具：安全 JSON 解析（带兜底） ===============
def _safe_json_extract(text: str) -> dict:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {}
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


# =============== 文本预处理 & 简易分词 ===============
_WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

def _normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return s

def _tokenize(s: str) -> List[str]:
    s = _normalize(s).lower()
    return _WORD_RE.findall(s)


# =============== 轻量 BM25（对 sessions 文本做检索） ===============
class BM25Sessions:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[int] = []
        self.doc_tokens: List[List[str]] = []
        self.df: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N: int = 0
        self.dl: List[int] = []

    def build(self, documents: List[Tuple[int, str]]):
        """
        documents: [(session_id, text), ...]
        """
        self.doc_ids.clear()
        self.doc_tokens.clear()
        self.df.clear()
        self.idf.clear()
        self.dl.clear()

        for sid, text in documents:
            toks = _tokenize(text or "")
            self.doc_ids.append(sid)
            self.doc_tokens.append(toks)
            self.dl.append(len(toks))
            for term in set(toks):
                self.df[term] += 1

        self.N = len(self.doc_ids)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0

        # 经典 BM25 idf
        for term, dfi in self.df.items():
            # 加 0.5 做平滑
            self.idf[term] = math.log(1 + (self.N - dfi + 0.5) / (dfi + 0.5))

    def _score_query_doc(self, q_terms: List[str], doc_idx: int) -> float:
        if not q_terms or doc_idx >= len(self.doc_tokens):
            return 0.0
        toks = self.doc_tokens[doc_idx]
        dl = self.dl[doc_idx] if self.dl else 0
        tf = Counter(toks)
        score = 0.0
        for t in q_terms:
            if t not in self.idf:
                continue
            f = tf.get(t, 0)
            if f == 0:
                continue
            idf = self.idf[t]
            denom = f + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
            score += idf * (f * (self.k1 + 1)) / (denom + 1e-9)
        return score

    def search(self, query_terms: List[str], topk: int = 10) -> List[int]:
        if not self.N or not query_terms:
            return []
        q_terms = [t for t in query_terms if t in self.idf]
        if not q_terms:
            # 若所有 query 词在索引外，直接返回空
            return []
        scored = []
        for i in range(self.N):
            s = self._score_query_doc(q_terms, i)
            if s > 0:
                scored.append((s, self.doc_ids[i]))
        scored.sort(reverse=True)
        return [sid for s, sid in scored[:topk]]

# ------------------------------
# MemoryAgent
# ------------------------------

class MemoryAgent:

    def __init__(self, llm, temperature=0.3):
        self.llm = llm
        self.memory_history = []
        self.system_prompt = MemoryAgent_PROMPT
        self.temperature = temperature


    def _next_event_id(self, prev_events: List[Dict], offset: int) -> str:
        existing_ids = [e.get("id", "") for e in prev_events if "id" in e]
        return f"event-{len(existing_ids) + offset + 1}"


    def memory_agent_step_merge(self, *, session_id: int, session_text: str,
                                prev_state: Optional[str] = None) -> str:



        prev_state_json = {"events": []} if not prev_state else json.loads(prev_state)
        historical_events: List[Dict] = prev_state_json.get("events", [])


        user_payload = {
            # "prev_events": historical_events,
            "session": session_text,
            "round_id": session_id
        }
        prompt = (
            self.system_prompt
            + "\n\n---\nUSER PAYLOAD (strict JSON expected):\n"
            + json.dumps(user_payload, ensure_ascii=False)
            + "\n\nReturn ONLY one valid JSON object as specified. No markdown, no extra text."
        )

        max_retry = 3

        for retry_count in range(max_retry + 1):
            try:
                output = self.llm.generate(str(prompt))
                output_json = json.loads(output)
                break
            except Exception as e:
                if retry_count < max_retry:
                    print(f"[MemoryAgent] 第 {session_id} 个 session 的 LLM 输出解析失败 (重试 {retry_count + 1}/{max_retry}): {str(e)}")
                else:
                    print(f"[MemoryAgent] 警告: 第 {session_id} 个 session 的 LLM 输出解析失败，已达到最大重试次数 {max_retry}，使用空 events")
                    output_json = {"events": []}

        new_events: List[Dict] = output_json.get("events", []) or []
        merged = list(historical_events)
        for i, ev in enumerate(new_events):
            if "id" not in ev:
                ev["id"] = self._next_event_id(historical_events, i)
            merged.append(ev)

        current_state_json = {"events": merged}
        current_state_str = json.dumps(current_state_json, ensure_ascii=False, indent=2)


        self.memory_history.append({
            "session_id": session_id,
            "session_text": session_text,
            "output": current_state_str
        })
        return current_state_str


    def run_memory_agent(self, sessions: List[str]) -> List[Dict]:
        """
        顺序处理所有 sessions，逐步演化记忆状态。
        """
        if not sessions:
            return []

        prev_state = None
        for i, sess in tqdm(enumerate(sessions, start=1), total=len(sessions), desc="处理 sessions", unit="段"):
            prev_state = self.memory_agent_step_merge(
                session_id=i,
                session_text=sess,
                prev_state=prev_state
            )
        return self.memory_history


    def get_final_memory_output(self) -> dict:
        if not self.memory_history:
            return {"events": []}
        last_out = self.memory_history[-1].get("output")
        try:
            return json.loads(last_out) if isinstance(last_out, str) else (last_out or {"events": []})
        except Exception:
            return {"events": []}




class DeepResearchAgent:
    """
    迭代流程（基于“原始 sessions”）：
      Planning（need_search? mode? session_ids? keywords?）
        → Searching（by_session &/or by_keywords；只针对 SESSIONS）
        → Reflection（够不够？不够就生成下一轮 plan）
        → Summarizing（结构化输出）
    """
    def __init__(self, llm, temperature=0.3, max_iterations: int = 3, bm25_k1: float = 1.5, bm25_b: float = 0.75):
        self.llm = llm
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.search_index = BM25Sessions(k1=bm25_k1, b=bm25_b)
        self._index_built = False
        self._session_map: Dict[int, str] = {}
        self.working_notes: List[Dict] = []  # [{"claim": "...", "support": [{"session_id": x, "quote": "..."}]}]
        self.decided_useful_sessions: List[int] = []  # 汇总 keep 的 session_id


    def _build_index_from_sessions(self, sessions: list):
        docs: List[Tuple[int, str]] = []
        self._session_map.clear()        

        if not sessions:
            self.search_index.build(docs)
            self._index_built = True
            return


        if sessions and isinstance(sessions[0], dict):
            for item in sessions:
                sid = int(item.get("session_id"))
                txt = item.get("session_text", "") or item.get("text", "")
                if not txt:
                    continue
                self._session_map[sid] = txt
                docs.append((sid, txt))
        else:
            for i, txt in enumerate(sessions, start=1):
                if not txt:
                    continue
                self._session_map[i] = txt
                docs.append((i, txt))

        self.search_index.build(docs)
        self._index_built = True
     


    def _planning_gate(self, question: str, memory: str = "", working_notes: Optional[List[Dict]] = None) -> dict:
        from prompts import PLANNING_DEEP_RESEARCH_PROMPT
        prompt = (PLANNING_DEEP_RESEARCH_PROMPT
                  + "\nQUESTION:\n" + str(question)
                  + "\nMEMORY:\n" + (memory or "")
                  + "\nWORKING_NOTES:\n" + json.dumps(working_notes or [], ensure_ascii=False))
        raw = self.llm.generate(prompt)
        data = _safe_json_extract(raw)
        if not isinstance(data, dict):
            data = {
                "need_search": True,
                "mode": "by_keywords",
                "session_ids": [],
                "keywords": [],
                "reasoning": "fallback"
            }
        mode = data.get("mode") or "by_keywords"
        if mode not in ("by_session", "by_keywords", "hybrid"):
            mode = "by_keywords"

        session_ids = []
        for x in (data.get("session_ids") or []):
            try:
                session_ids.append(int(x))
            except Exception:
                continue
        keywords = [str(x) for x in (data.get("keywords") or []) if str(x).strip()]

        return {
            "need_search": bool(data.get("need_search", True)),
            "mode": mode,
            "session_ids": session_ids[:12],
            "keywords": keywords[:8],
            "reasoning": data.get("reasoning", "")
        }


    # ---------- Searching（对“sessions”做检索） ----------
    def _search_by_session_ids(self, session_ids: List[int]) -> List[int]:
        found = []
        for sid in session_ids or []:
            if sid in self._session_map and self._session_map[sid]:
                found.append(sid)

        seen, ordered = set(), []
        for s in found:
            if s not in seen:
                seen.add(s); ordered.append(s)
        return ordered


    def _search_by_keywords(self, keywords: List[str], topk_sessions: int = 8) -> List[int]:
        if not self._index_built:
            return []
        q_terms = []
        for kw in keywords or []:
            q_terms.extend(_tokenize(kw))
        q_terms = list(dict.fromkeys(q_terms))[:12]
        if not q_terms:
            return []
        return self.search_index.search(q_terms, topk=topk_sessions)



    def _search_router(self, plan: dict, topk_sessions: int = 8) -> List[int]:
        results: List[int] = []
        mode = plan.get("mode")
        if mode in ("by_session", "hybrid"):
            results += self._search_by_session_ids(plan.get("session_ids", []))
        if mode in ("by_keywords", "hybrid"):
            results += self._search_by_keywords(plan.get("keywords", []), topk_sessions=topk_sessions)

        seen, merged = set(), []
        for sid in results:
            if sid not in seen:
                seen.add(sid); merged.append(sid)
        return merged


    def _reflection(self, memory: str, question: str, ctx_session_ids: List[int], working_notes: List[Dict]) -> Tuple[bool, dict]:
        from prompts import REPLAN_FROM_SESSIONS_PROMPT
        snippets = []
        for sid in ctx_session_ids[:6]:
            txt = self._session_map.get(sid, "") or ""
            snippets.append({"session_id": sid, "snippet": txt})

        prompt = (REPLAN_FROM_SESSIONS_PROMPT
                  + "\nQUESTION:\n" + str(question)
                  + "\nMEMORY:\n" + str(memory)
                  + "\nWORKING_NOTES:\n" + json.dumps(working_notes or [], ensure_ascii=False)
                  + "\nSESSIONS_SNIPPETS:\n" + json.dumps(snippets, ensure_ascii=False))
        raw = self.llm.generate(prompt)
        data = _safe_json_extract(raw)

        if not isinstance(data, dict):
            return False, {
                "plan": {"need_search": True, "mode": "by_keywords", "session_ids": [], "keywords": [], "reasoning": "fallback"},
                "notes_delta": [],
                "keep_ids": []
            }

        enough = bool(data.get("enough", False))
        next_plan = data.get("next_plan") or {}
        notes_delta = data.get("notes_delta") or []
        keep_ids_raw = data.get("keep_session_ids") or []


        mode = next_plan.get("mode") or "by_keywords"
        if mode not in ("by_session", "by_keywords", "hybrid"):
            mode = "by_keywords"

        session_ids = []
        for x in (next_plan.get("session_ids") or []):
            try:
                session_ids.append(int(x))
            except Exception:
                continue
        keywords = [str(x) for x in (next_plan.get("keywords") or []) if str(x).strip()]

        keep_ids: List[int] = []
        for x in keep_ids_raw:
            try:
                keep_ids.append(int(x))
            except Exception:
                continue


        plan_norm = {
            "need_search": not enough,
            "mode": mode,
            "session_ids": session_ids[:12],
            "keywords": keywords[:8],
            "reasoning": "reflection-driven"
        }
        return enough, {"plan": plan_norm, "notes_delta": notes_delta, "keep_ids": keep_ids}


    def _summarize_sessions(self, question: str, session_ids: List[int]) -> dict:
        from prompts import SESSION_SUMMARY_PROMPT
        sessions_data = []
        for sid in session_ids:
            txt = self._session_map.get(sid, "")
            if txt:
                sessions_data.append({"session_id": sid, "text": txt})

        prompt = (SESSION_SUMMARY_PROMPT
                  + "\nQUESTION:\n" + str(question)
                  + "\nRETRIEVED_SESSIONS:\n" + json.dumps(sessions_data, ensure_ascii=False))
        raw = self.llm.generate(prompt)
        data = _safe_json_extract(raw)

        if isinstance(data, dict):
            return {
                "summary": data.get("summary", ""),
                "session_ids_used": [s["session_id"] for s in sessions_data]
            }
        return {
            "summary": raw if isinstance(raw, str) else "",
            "session_ids_used": [s["session_id"] for s in sessions_data]
        }



    def _summarize_memory(self, question: str, memory: str) -> dict:
        from prompts import MEMORY_SUMMARY_PROMPT
        prompt = (MEMORY_SUMMARY_PROMPT
                  + "\nQUESTION:\n" + str(question)
                  + "\nMEMORY:\n" + str(memory))
        raw = self.llm.generate(prompt)
        data = _safe_json_extract(raw)

        if isinstance(data, dict):
            return {
                "summary": data.get("summary", ""),
                "session_ids_used": []
            }
        return {
            "summary": raw if isinstance(raw, str) else "",
            "session_ids_used": []
        }

    @staticmethod
    def _dedupe_notes(notes: List[Dict]) -> List[Dict]:
        seen = set()
        deduped = []
        for n in notes or []:
            claim = (n.get("claim") or "").strip()
            if not claim:
                continue
            if claim not in seen:
                seen.add(claim)
                # 规范 support
                sup = n.get("support") or []
                clean_sup = []
                for s in sup:
                    try:
                        clean_sup.append({
                            "session_id": int(s.get("session_id")),
                            "quote": (s.get("quote") or "").strip()[:240]
                        })
                    except Exception:
                        continue
                deduped.append({"claim": claim, "support": clean_sup})
        return deduped

    @staticmethod
    def _merge_summary_with_notes(summary_json_str: str, notes: List[Dict]) -> str:
        """
        把 working_notes 中未体现的 claim 以“补充证据”形式并入 JSON summary 字段（不改变外层 schema）。
        - 若 summary_json_str 是纯字符串，将其包一层“正文 + 补充”。
        - 若是 JSON 字符串（仅含 "summary"），则合并为同一字符串。
        """
        try:
            base_summary = json.loads(summary_json_str)  # 若本身就是 JSON，就不动
            # 极少见；通常你的 SESSION_SUMMARY_PROMPT 返回 {"summary": "..."}，此处 summary_json_str 其实是 str
            return summary_json_str
        except Exception:
            pass

        notes = DeepResearchAgent._dedupe_notes(notes)
        if not notes:
            return summary_json_str

        # 生成补充段
        extra_lines = ["\n\n[Supplemental Evidence]"]
        for i, n in enumerate(notes, 1):
            sup_strs = []
            for s in n.get("support", []):
                sup_strs.append(f"(s{int(s['session_id'])})")
            sup_join = " ".join(sup_strs) if sup_strs else ""
            extra_lines.append(f"{i}. {n.get('claim','')}{(' ' + sup_join) if sup_join else ''}")
        merged = (summary_json_str or "").rstrip() + "\n" + "\n".join(extra_lines)
        return merged


    def deep_research(self, question: str, memory: dict, sessions: List[Dict], max_sessions: int = 8) -> dict:
        """
        参数：
          - question: 用户问题
          - memory: MemoryAgent 最终状态（dict 或 str）
          - sessions: [{"session_id": int, "session_text": str}, ...] 或 纯文本列表
        返回：
          {
            "summary": str,
            "session_ids_used": [...],
            "iterations": int,
            "working_notes": [...],            # 新增：返回累积的笔记，便于调试 / 透明化
            "decided_useful_sessions": [...]   # 新增：最终保留的上下文
          }
        """
        # 1) 索引构建
        memory_str = memory if isinstance(memory, str) else json.dumps(memory or {}, ensure_ascii=False)
        
        self._build_index_from_sessions(sessions)

        # 2) 初次 Planning
        plan = self._planning_gate(question, memory_str, self.working_notes)
        context_sessions: List[int] = []


        if not plan.get("need_search"):
            result = self._summarize_memory(question, memory_str)
            result["summary"] = self._merge_summary_with_notes(result["summary"], self.working_notes)
            result["iterations"] = 0
            result["working_notes"] = list(self.working_notes)
            result["decided_useful_sessions"] = list(self.decided_useful_sessions)
            return result

        # 3) 循环：Searching → Reflection →（足够则 Summarizing）
        for it in range(self.max_iterations):
            new_sessions = self._search_router(plan, topk_sessions=max_sessions)
            
            seen = set(context_sessions)
            for sid in new_sessions:
                if sid not in seen:
                    context_sessions.append(sid)
                    seen.add(sid)
            context_sessions = context_sessions[:max_sessions]

            enough, ref_out = self._reflection(memory_str, question, context_sessions, self.working_notes)

            self.working_notes.extend(ref_out.get("notes_delta", []))
            self.working_notes = self._dedupe_notes(self.working_notes)

            keep_ids = ref_out.get("keep_ids", [])
            if keep_ids:
                # 记录全局“已确认有用”
                for k in keep_ids:
                    if k not in self.decided_useful_sessions:
                        self.decided_useful_sessions.append(k)
                # 把 keep 放前面，再追加其它 context
                front = []
                seen = set()
                for sid in keep_ids + context_sessions:
                    if sid not in seen:
                        front.append(sid); seen.add(sid)
                context_sessions = front[:max_sessions]            


            if enough:
                result = self._summarize_sessions(question, context_sessions)
                result["summary"] = self._merge_summary_with_notes(result["summary"], self.working_notes)
                result["iterations"] = it + 1
                result["working_notes"] = list(self.working_notes)
                result["decided_useful_sessions"] = list(self.decided_useful_sessions)
                return result

            plan = ref_out["plan"]

        result = self._summarize_sessions(question, context_sessions)
        result["summary"] = self._merge_summary_with_notes(result["summary"], self.working_notes)
        result["iterations"] = self.max_iterations
        result["working_notes"] = list(self.working_notes)
        result["decided_useful_sessions"] = list(self.decided_useful_sessions)
        return result
