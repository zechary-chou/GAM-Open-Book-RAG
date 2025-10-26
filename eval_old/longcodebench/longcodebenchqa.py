#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LongCodeBench QA with Memory Agent Architecture (v3)

基于记忆处理架构的LongCodeBench问答系统，支持三种模式：
- memory_only: 仅构建记忆
- answer_only: 仅回答问题（需要已有记忆）
- build_and_answer: 构建记忆并回答问题

支持GPU并行处理，将模型分布在8个GPU上串行运行。
"""

import os
import re
import json
import argparse
import time
import sys
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from tqdm import tqdm
import threading
import queue

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import tiktoken
from gam.llm_call import OpenRouterModel
from gam.agents import (
    MemoryAgent, 
    DeepResearchAgent
)
from gam.utils import (
    build_session_chunks_from_text,
    build_pages_from_sessions_and_abstracts,
    safe_json_extract
)


# ========== 工具函数 ==========
def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()




def ensure_dir(p: str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ========== LongCodeBench 数据处理 ==========
def load_bucket(path: str) -> List[Dict[str, Any]]:
    """
    Load either:
      - JSON array file
      - JSONL file (one obj per line)
    """
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if not txt:
            return []
        # JSONL heuristic: first non-space char is '{' but not '[' and contains multiple lines
        if txt[0] != '[':
            # try JSONL
            items = []
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
            return items
        else:
            return json.loads(txt)


def normalize_gold_letter(v: Any) -> str:
    """
    Accept 'a'/'b'/'c'/'d' or 'A'..'D' (and tolerate strings like 'correct=C').
    """
    if isinstance(v, str):
        m = re.search(r"[ABCD]", v, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()
    raise ValueError(f"gold letter malformed: {v}")


def parse_final_answer(text: str) -> Optional[str]:
    """
    Prefer 'Final Answer: X'. Fallback: first standalone A-D.
    """
    if not text:
        return None
    m = re.search(r"Final\s*Answer\s*:\s*([ABCD])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\b([ABCD])\b", text, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return None


def truncate_tokens(s: str, max_tokens: Optional[int], model: str) -> str:
    """
    Truncate text based on token count, similar to pred.py logic.
    If text exceeds max_tokens, keep first half and last half.
    """
    if max_tokens is None:
        return s
    
    tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    # For OpenAI models, use tiktoken
    input_ids = tokenizer.encode(s, disallowed_special=())
    if len(input_ids) <= max_tokens:
        return s
        # Take first half and last half
    half_tokens = max_tokens // 2
    truncated_ids = input_ids[:half_tokens] + input_ids[-half_tokens:]
    return tokenizer.decode(truncated_ids)


# ========== 会话块构建 ==========
def build_session_chunks_for_sample(sample: Dict[str, Any], model_name: str = "gpt-4o-mini", max_tokens: int = 2000) -> List[str]:
    """
    为LongCodeBench样本构建会话块
    使用通用的build_session_chunks_from_text函数
    
    Args:
        sample: 包含repo_text的样本数据
        model_name: OpenAI模型名称，用于选择对应的tokenizer
        max_tokens: 每个会话块的最大token数量
    """
    repo_text = sample.get("repo_text") or ""
    return build_session_chunks_from_text(repo_text, max_tokens, model_name)


# ========== 记忆处理 ==========
def build_memory_for_sample(llm, session_chunks: List[str], temperature: float = 0.3):
    """
    顺序运行 MemoryAgent：只负责把 sessions -> events state + abstracts。
    返回：(mem_agent, memory_history, final_memory_state, session_abstracts)
    """
    mem = MemoryAgent(llm, temperature=temperature)
    history = mem.run_memory_agent(sessions=session_chunks)
    
    # 使用新的get_memory_with_abstracts方法
    memory_with_abstracts = mem.get_memory_with_abstracts()
    final_state = memory_with_abstracts
    session_abstracts = memory_with_abstracts.get('session_abstracts', [])
    
    return mem, history, final_state, session_abstracts




def memory_deep_research(qs_llm, final_memory: Dict[str, Any],
                              session_chunks: List[str],
                              question: str,
                              session_abstracts: List[str] = None,
                              max_sessions: int = 6,
                              temperature: float = 0.3) -> Dict[str, Any]:
    """
    只回答：不重建记忆。要求已经有 final_memory（可从磁盘加载），且能拿到原始 session 文本。
    现在支持使用session_abstracts来创建pages进行检索。
    """
    # 构建pages
    if session_abstracts:
        pages = build_pages_from_sessions_and_abstracts(session_chunks, session_abstracts)
    else:
        # 如果没有abstracts，直接使用sessions作为pages
        pages = session_chunks
    
    # 使用DeepResearchAgent进行research
    ret = DeepResearchAgent(qs_llm, temperature=temperature)
    result = ret.deep_research(question, final_memory, pages, max_sessions)
    
    return result


def deep_research_answer(qs_llm, final_memory, session_chunks: List[str],
                         question: str, session_abstracts: List[str] = None,
                         max_sessions: int = 6, temperature: float = 0.3) -> Tuple[str, List[int]]:
    """
    用 DeepResearchAgent 取回相关 sessions + 总结。
    返回：(summary_text, used_session_ids)
    """
    result = memory_deep_research(
        qs_llm, final_memory, session_chunks, question, 
        session_abstracts=session_abstracts,
        max_sessions=max_sessions, temperature=temperature
    )
    session_ids = result.get("session_ids_used") or result.get("session_ids") or []
    summary = result.get("summary", "")

    return summary, session_ids


# ========== 问答处理 ==========
def make_memory_only_prompt(prompt_goal: str, memory_obj: Any, question: str) -> str:
    """
    构建基于记忆的提示词：prompt_goal + memory + question
    """
    mem_str = json.dumps(memory_obj, ensure_ascii=False, indent=2) if isinstance(memory_obj, dict) else str(memory_obj)
    return f"""{prompt_goal}

MEMORY STATE:
{mem_str}

QUESTION:
{question}

Answer:"""


def make_summary_prompt(prompt_goal: str, summary: str, question: str) -> str:
    """
    构建基于总结的提示词：prompt_goal + summary + question
    """
    return f"""{prompt_goal}

SUMMARY:
{summary}

QUESTION:
{question}

Answer:"""


def answer_with_summary(qs_llm, prompt_goal: str, summary: str, question: str, temperature: float = 0.3) -> str:
    prompt = make_summary_prompt(prompt_goal, summary, question)
    raw = qs_llm.generate(prompt, temperature=temperature, max_tokens=1024)
    return raw


def answer_with_memory(qs_llm, prompt_goal: str, final_memory: Dict[str, Any], question: str, temperature: float = 0.3) -> str:
    prompt = make_memory_only_prompt(prompt_goal, final_memory, question)
    raw = qs_llm.generate(prompt, temperature=temperature, max_tokens=1024)
    return raw


# ========== 记忆文件管理 ==========
# 删除不再使用的load_existing_memory函数，因为现在使用全局文件


# 删除不再使用的load_existing_qa_results函数，因为现在使用全局文件


def is_sample_processed(sample_id: str, results_dir: str, mode: str) -> bool:
    """检查样本是否已经被处理过"""
    if mode == "memory_only":
        memory_file = os.path.join(results_dir, "all_samples_memory.jsonl")
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            if data.get("sample_id") == sample_id:
                                return True
            except Exception as e:
                print(f"Warning: 无法读取记忆文件: {e}")
        return False
    elif mode in ["answer_only", "build_and_answer"]:
        qa_file = os.path.join(results_dir, "all_samples_qa.jsonl")
        if os.path.exists(qa_file):
            try:
                with open(qa_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            if data.get("sample_id") == sample_id:
                                return True
            except Exception as e:
                print(f"Warning: 无法读取问答文件: {e}")
        return False
    return False



# ========== 处理函数 ==========
def process_memory_only(sample: Dict[str, Any], sample_index: int, gpu_id: int, memory_model, args) -> Dict[str, Any]:
    """处理memory_only模式"""
    try:
        sample_id = (sample.get("id") or sample.get("_id") or sample.get("uid") or
               md5((sample.get("repo") or "") + (sample.get("question") or "") + str(sample_index)))
        print(f"[Sample-{sample_index}] Processing {sample_id}")
        
        # 检查是否已处理
        if is_sample_processed(sample_id, args.outdir, "memory_only"):
            print(f"[Sample-{sample_index}] [skip] 样本已处理过")
            return {"sample_id": sample_id, "sample_index": sample_index, "status": "skipped"}
        
        # 构建会话块
        try:
            session_chunks = build_session_chunks_for_sample(sample, args.model_memory, args.max_tokens)
        except Exception as e:
            print(f"[Sample-{sample_index}] Warning: 无法使用tiktoken切分: {e}")
            # 使用字符切分作为fallback
            from gam.utils import _build_chunks_by_char
            session_chunks = _build_chunks_by_char(sample.get("repo_text", ""), args.max_tokens * 4)
        
        # 构建记忆
        mem_agent, history, final_memory, session_abstracts = build_memory_for_sample(memory_model, session_chunks, args.temperature)
        
        # 立即保存
        
        memory_result = {
            "sample_id": sample_id,
            "sample_index": sample_index,
            # "memory_history": history,
            "final_memory": final_memory,
            "session_abstracts": session_abstracts,
            "num_sessions": len(session_chunks),
            "mode": "memory_only",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        memory_file = os.path.join(args.outdir, "all_samples_memory.jsonl")
        with open(memory_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(memory_result, ensure_ascii=False) + "\n")
        
        print(f"[Sample-{sample_index}] Memory saved")
        return {"sample_id": sample_id, "sample_index": sample_index, "num_sessions": len(session_chunks), "status": "success"}
        
    except Exception as e:
        print(f"[Sample-{sample_index}] ERROR: {e}")
        return {"sample_id": sample.get("id", f"sample-{sample_index}"), "sample_index": sample_index, "status": "error", "error": str(e)}


def process_answer_only(sample: Dict[str, Any], sample_index: int, gpu_id: int, research_model, answer_model, args) -> Dict[str, Any]:
    """处理answer_only模式"""
    try:
        sample_id = (sample.get("id") or sample.get("_id") or sample.get("uid") or
               md5((sample.get("repo") or "") + (sample.get("question") or "") + str(sample_index)))
        print(f"[Sample-{sample_index}] Processing {sample_id}")
        
        # 检查是否已处理
        if is_sample_processed(sample_id, args.outdir, "answer_only"):
            print(f"[Sample-{sample_index}] [skip] 样本已处理过")
            return {"sample_id": sample_id, "sample_index": sample_index, "status": "skipped"}
        
        # 加载记忆
        memory_file = os.path.join(args.outdir, "all_samples_memory.jsonl")
        final_memory = None
        session_abstracts = None
        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            memory_data = json.loads(line.strip())
                            if memory_data.get("sample_id") == sample_id:
                                final_memory = memory_data.get("final_memory")
                                session_abstracts = memory_data.get("session_abstracts", [])
                                break
            except Exception as e:
                print(f"[Sample-{sample_index}] Warning: 无法加载记忆: {e}")
        
        if final_memory is None:
            print(f"[Sample-{sample_index}] [error] 未找到记忆")
            return {"sample_id": sample_id, "sample_index": sample_index, "status": "error", "error": "No memory found"}
        
        # 构建会话块
        try:
            session_chunks = build_session_chunks_for_sample(sample, args.model_research, args.max_tokens)
        except Exception as e:
            print(f"[Sample-{sample_index}] Warning: 无法使用tiktoken切分: {e}")
            # 使用字符切分作为fallback
            from gam.utils import _build_chunks_by_char
            session_chunks = _build_chunks_by_char(sample.get("repo_text", ""), args.max_tokens * 4)
        
        # 处理问答
        question = sample.get("question", "")
        gold_answer = sample.get("correct_letter", "")
        prompt_goal = sample.get("prompt_goal", "")
        
        if not question or not prompt_goal:
            return {"sample_id": sample_id, "sample_index": sample_index, "status": "error", "error": "Missing question or prompt_goal"}
        
        final_memory_str = json.dumps(final_memory, ensure_ascii=False, indent=2)
        summary, session_ids = deep_research_answer(research_model, final_memory_str, session_chunks, question, 
                                                   session_abstracts=session_abstracts, max_sessions=args.max_sessions, temperature=args.temperature)
        
        summary_answer = answer_with_summary(answer_model, prompt_goal, summary, question, args.temperature)
        memory_answer = answer_with_memory(answer_model, prompt_goal, final_memory_str, question, args.temperature)
        
        pred_summary = parse_final_answer(summary_answer)
        pred_memory = parse_final_answer(memory_answer)
        is_ok_summary = bool(pred_summary) and (pred_summary.upper() == gold_answer.upper())
        is_ok_memory = bool(pred_memory) and (pred_memory.upper() == gold_answer.upper())
        
        # 立即保存
        result = {
            "sample_id": sample_id,
            "sample_index": sample_index,
            "question": question,
            "gold_answer": gold_answer,
            "memory_answer": memory_answer,
            "summary": summary,
            "summary_answer": summary_answer,
            "pred_summary": pred_summary,
            "pred_memory": pred_memory,
            "correct_summary": is_ok_summary,
            "correct_memory": is_ok_memory,
            "retrieved_session_ids": session_ids,
            "mode": "answer_only",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        qa_file = os.path.join(args.outdir, "all_samples_qa.jsonl")
        with open(qa_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"[Sample-{sample_index}] QA result saved")
        return {"sample_id": sample_id, "sample_index": sample_index, "processed_questions": 1, "status": "success"}
        
    except Exception as e:
        print(f"[Sample-{sample_index}] ERROR: {e}")
        return {"sample_id": sample.get("id", f"sample-{sample_index}"), "sample_index": sample_index, "status": "error", "error": str(e)}


def process_build_and_answer(sample: Dict[str, Any], sample_index: int, gpu_id: int, memory_model, research_model, answer_model, args) -> Dict[str, Any]:
    """处理build_and_answer模式"""
    try:
        sample_id = (sample.get("id") or sample.get("_id") or sample.get("uid") or
               md5((sample.get("repo") or "") + (sample.get("question") or "") + str(sample_index)))
        print(f"[Sample-{sample_index}] Processing {sample_id}")
        
        # 检查是否已处理
        if is_sample_processed(sample_id, args.outdir, "build_and_answer"):
            print(f"[Sample-{sample_index}] [skip] 样本已处理过")
            return {"sample_id": sample_id, "sample_index": sample_index, "status": "skipped"}
        
        # 构建会话块
        try:
            session_chunks = build_session_chunks_for_sample(sample, args.model_memory, args.max_tokens)
        except Exception as e:
            print(f"[Sample-{sample_index}] Warning: 无法使用tiktoken切分: {e}")
            # 使用字符切分作为fallback
            from gam.utils import _build_chunks_by_char
            session_chunks = _build_chunks_by_char(sample.get("repo_text", ""), args.max_tokens * 4)
        
        # 构建记忆
        mem_agent, history, final_memory, session_abstracts = build_memory_for_sample(memory_model, session_chunks, args.temperature)
        
        # 保存记忆
        memory_result = {
            "sample_id": sample_id,
            "sample_index": sample_index,
            # "memory_history": history,
            "final_memory": final_memory,
            "session_abstracts": session_abstracts,
            "num_sessions": len(session_chunks),
            "mode": "build_and_answer",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        memory_file = os.path.join(args.outdir, "all_samples_memory.jsonl")
        with open(memory_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(memory_result, ensure_ascii=False) + "\n")
        
        # 处理问答
        question = sample.get("question", "")
        gold_answer = sample.get("correct_letter", "")
        prompt_goal = sample.get("prompt_goal", "")
        
        if not question or not prompt_goal:
            return {"sample_id": sample_id, "sample_index": sample_index, "status": "error", "error": "Missing question or prompt_goal"}
        
        final_memory_str = json.dumps(final_memory, ensure_ascii=False, indent=2)
        summary, session_ids = deep_research_answer(research_model, final_memory_str, session_chunks, question, 
                                                   session_abstracts=session_abstracts, max_sessions=args.max_sessions, temperature=args.temperature)
        
        summary_answer = answer_with_summary(answer_model, prompt_goal, summary, question, args.temperature)
        memory_answer = answer_with_memory(answer_model, prompt_goal, final_memory_str, question, args.temperature)
        
        pred_summary = parse_final_answer(summary_answer)
        pred_memory = parse_final_answer(memory_answer)
        is_ok_summary = bool(pred_summary) and (pred_summary.upper() == gold_answer.upper())
        is_ok_memory = bool(pred_memory) and (pred_memory.upper() == gold_answer.upper())
        
        # 保存问答结果
        result = {
            "sample_id": sample_id,
            "sample_index": sample_index,
            "question": question,
            "gold_answer": gold_answer,
            "memory_answer": memory_answer,
            "summary": summary,
            "summary_answer": summary_answer,
            "pred_summary": pred_summary,
            "pred_memory": pred_memory,
            "correct_summary": is_ok_summary,
            "correct_memory": is_ok_memory,
            "retrieved_session_ids": session_ids,
            "mode": "build_and_answer",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        qa_file = os.path.join(args.outdir, "all_samples_qa.jsonl")
        with open(qa_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"[Sample-{sample_index}] Results saved")
        return {"sample_id": sample_id, "sample_index": sample_index, "num_sessions": len(session_chunks), "processed_questions": 1, "status": "success"}
        
    except Exception as e:
        print(f"[Sample-{sample_index}] ERROR: {e}")
        return {"sample_id": sample.get("id", f"sample-{sample_index}"), "sample_index": sample_index, "status": "error", "error": str(e)}


# ========== 全局记忆文件生成 ==========
def generate_global_memory_files(results_dir: str, all_stats: List[Dict[str, Any]]):
    """
    生成全局记忆文件：从JSONL文件中读取并生成汇总文件
    """
    print("\n== 生成全局记忆文件 ==")
    
    memory_file = os.path.join(results_dir, "all_samples_memory.jsonl")
    
    if not os.path.exists(memory_file):
        print(f"记忆文件不存在: {memory_file}")
        return
    
    all_memory_histories = []
    all_final_memories = []
    
    # 从JSONL文件中读取记忆数据
    try:
        with open(memory_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    memory_data = json.loads(line.strip())
                    sample_id = memory_data.get("sample_id")
                    memory_history = memory_data.get("memory_history")
                    final_memory = memory_data.get("final_memory")
                    
                    if sample_id and memory_history:
                        all_memory_histories.append({
                            "sample_id": sample_id,
                            "memory_history": memory_history
                        })
                    
                    if sample_id and final_memory:
                        all_final_memories.append({
                            "sample_id": sample_id,
                            "final_memory": final_memory
                        })
    except Exception as e:
        print(f"[WARN] Failed to read memory file: {e}")
        return
    
    # 保存全局记忆文件
    global_memory_file = os.path.join(results_dir, "all_samples_memory_histories.json")
    global_final_memory_file = os.path.join(results_dir, "all_samples_final_memories.json")
    
    with open(global_memory_file, "w", encoding="utf-8") as f:
        json.dump(all_memory_histories, f, ensure_ascii=False, indent=2)
    print(f"全局记忆历史文件保存到: {global_memory_file}")
    
    with open(global_final_memory_file, "w", encoding="utf-8") as f:
        json.dump(all_final_memories, f, ensure_ascii=False, indent=2)
    print(f"全局最终记忆文件保存到: {global_final_memory_file}")
    
    print(f"总共处理了 {len(all_memory_histories)} 个样本的记忆历史")
    print(f"总共处理了 {len(all_final_memories)} 个样本的最终记忆")


# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="LongCodeBench QA with Memory Agent Architecture (v3)")
    parser.add_argument("--data", type=str, default=r"D:\python\Streamingllm\datasets\longcodebench\64K.json", help="Path to dataset")
    parser.add_argument("--mode", type=str, default="answer_only", choices=["memory_only", "answer_only", "build_and_answer"], help="Processing mode")
    parser.add_argument("--outdir", type=str, default=r"D:\python\general-agentic-memory\examples\longcodebench\results\64k", help="Output directory")
    parser.add_argument("--max-sessions", type=int, default=5, help="Max sessions retrieved per question")
    parser.add_argument("--sample-idx", type=int, default=-1, help="Run a single sample (0-based). -1 = run all")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate first N samples (0 = all)")
    parser.add_argument("--model-memory", type=str, default="gpt-4o-mini", help="Model for memory processing")
    parser.add_argument("--model-research", type=str, default="gpt-4o-mini", help="Model for research")
    parser.add_argument("--model-answer", type=str, default="gpt-4o-mini", help="Model for answering")
    parser.add_argument("--max-tokens", type=int, default=8096, help="Max tokens per session chunk")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep between requests (seconds)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing results")
    
    args = parser.parse_args()

    # 创建输出目录
    ensure_dir(args.outdir)

    # 载入数据
    data = load_bucket(args.data)
    if args.limit and args.limit > 0:
        data = data[:args.limit]
    
    if not isinstance(data, list) or not data:
        print("[Error] No samples found in dataset.")
        return

    # 选择范围
    indices = [args.sample_idx] if args.sample_idx >= 0 else list(range(len(data)))

    print(f"\n== LongCodeBench QA with Memory Agent (v3) ==")
    print(f"  模式: {args.mode}")
    print(f"  数据文件: {args.data}")
    print(f"  输出目录: {args.outdir}")
    print(f"  处理样本数: {len(indices)}")
    print(f"  最大会话数: {args.max_sessions}")

    all_stats = []
    start_time = time.time()

    # 初始化API模型
    print(f"\n== 初始化API模型 ==")
    
    if args.mode == "memory_only":
        print(f"初始化记忆模型: {args.model_memory}")
        memory_model = OpenRouterModel(
            model=args.model_memory,
            max_retries=3
        )
        print(f"记忆模型初始化完成: {args.model_memory}")
        
    elif args.mode == "answer_only":
        print(f"初始化研究模型: {args.model_research}")
        research_model = OpenRouterModel(
            model=args.model_research,
            max_retries=3
        )
        
        print(f"初始化回答模型: {args.model_answer}")
        answer_model = OpenRouterModel(
            model=args.model_answer,
            max_retries=3
        )
        
        print(f"研究模型初始化完成: {args.model_research}")
        print(f"回答模型初始化完成: {args.model_answer}")
        
    elif args.mode == "build_and_answer":
        print(f"初始化记忆模型: {args.model_memory}")
        memory_model = OpenRouterModel(
            model=args.model_memory,
            max_retries=3
        )
        
        print(f"初始化研究模型: {args.model_research}")
        research_model = OpenRouterModel(
            model=args.model_research,
            max_retries=3
        )
        
        print(f"初始化回答模型: {args.model_answer}")
        answer_model = OpenRouterModel(
            model=args.model_answer,
            max_retries=3
        )
        
        print(f"记忆模型初始化完成: {args.model_memory}")
        print(f"研究模型初始化完成: {args.model_research}")
        print(f"回答模型初始化完成: {args.model_answer}")
    
    print("== 模型初始化完成，开始处理样本 ==")
    
    # 串行处理样本
    for i, idx in enumerate(tqdm(indices, desc="处理样本", unit="样本")):
        if idx < 0 or idx >= len(data):
            print(f"[Warn] sample-idx {idx} out of range, skip.")
            continue

        sample = data[idx]
        
        # 根据模式处理样本
        if args.mode == "memory_only":
            stats = process_memory_only(sample, idx, 0, memory_model, args)
        elif args.mode == "answer_only":
            stats = process_answer_only(sample, idx, 0, research_model, answer_model, args)
        elif args.mode == "build_and_answer":
            stats = process_build_and_answer(sample, idx, 0, memory_model, research_model, answer_model, args)
        
        all_stats.append(stats)
        
        # 显示进度
        successful_samples = [s for s in all_stats if s.get('status') == 'success']
        failed_samples = [s for s in all_stats if s.get('status') == 'error']
        skipped_samples = [s for s in all_stats if s.get('status') == 'skipped']
        
        print(f"[Progress] 已处理: {len(all_stats)}/{len(indices)}, 成功: {len(successful_samples)}, 失败: {len(failed_samples)}, 跳过: {len(skipped_samples)}")
        
        if args.sleep > 0:
            time.sleep(args.sleep)

    end_time = time.time()
    processing_time = end_time - start_time

    # 生成全局记忆文件
    if args.mode in ["memory_only", "build_and_answer"]:
        generate_global_memory_files(args.outdir, all_stats)

    # 汇总统计
    if all_stats:
        successful_samples = [s for s in all_stats if s.get('status') == 'success']
        failed_samples = [s for s in all_stats if s.get('status') == 'error']
        skipped_samples = [s for s in all_stats if s.get('status') == 'skipped']
        
        total_samples = len(all_stats)
        total_sessions = sum(s.get("num_sessions", 0) for s in successful_samples)
        total_questions = sum(s.get("processed_questions", 0) for s in successful_samples)
        
        print("\n== 全局汇总 ==")
        print(f"  处理方式: API模型调用")
        print(f"  总样本数: {total_samples}")
        print(f"  成功样本: {len(successful_samples)}")
        print(f"  失败样本: {len(failed_samples)}")
        print(f"  跳过样本: {len(skipped_samples)}")
        print(f"  总会话数: {total_sessions}")
        print(f"  总问题数: {total_questions}")
        print(f"  处理时间: {processing_time:.2f} 秒")
        print(f"  平均每样本时间: {processing_time/total_samples:.2f} 秒")
        print(f"  结果保存到: {args.outdir}")
        
        # 显示失败的样本
        if failed_samples:
            print("\n== 失败样本 ==")
            for failed in failed_samples:
                print(f"  样本 {failed['sample_index']} ({failed['sample_id']}): {failed.get('error', '未知错误')}")
        
        # 保存全局统计信息
        global_stats = {
            "processing_method": "api_model_calls",
            "mode": args.mode,
            "total_samples": total_samples,
            "successful_samples": len(successful_samples),
            "failed_samples": len(failed_samples),
            "skipped_samples": len(skipped_samples),
            "total_sessions": total_sessions,
            "total_questions": total_questions,
            "processing_time_seconds": processing_time,
            "average_time_per_sample": processing_time/total_samples,
            "models_used": {
                "memory": args.model_memory,
                "research": args.model_research,
                "answer": args.model_answer
            },
            "per_sample_stats": all_stats
        }
        
        global_stats_file = os.path.join(args.outdir, "global_stats.json")
        with open(global_stats_file, 'w', encoding='utf-8') as f:
            json.dump(global_stats, f, ensure_ascii=False, indent=2)
        print(f"  全局统计保存到: {global_stats_file}")

    print(f"\n处理完成！结果保存在: {args.outdir}")


if __name__ == "__main__":
    main()
