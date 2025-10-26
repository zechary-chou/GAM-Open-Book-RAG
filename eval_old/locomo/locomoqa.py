import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

# === Project-local imports ===
from gam.agents import (
    MemoryAgent, 
    DeepResearchAgent
)
from gam.utils import (
    build_session_chunks_from_text,
    build_pages_from_sessions_and_abstracts
)
from gam.llm_call import OpenRouterModel   # 使用 OpenRouterModel（你提供的）  # noqa: E402



max_questions_demo = None  # None 表示全部


# ========== 工具函数 ==========
def safe_json_extract(candidate: Any) -> Optional[Dict[str, Any]]:
    """尽量把模型输出（string/dict）解析成 dict，失败返回 None。"""
    if isinstance(candidate, dict):
        return candidate
    if not isinstance(candidate, str):
        return None
    s = candidate.strip()
    l = s.find('{')
    r = s.rfind('}')
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(s[l:r+1])
    except Exception:
        return None


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


# ========== LoCoMo 数据抽取：保持你原有的结构与字段适配 ==========
def load_locomo(json_path: str) -> List[Dict[str, Any]]:
    """Load LoCoMo JSON and return the list of samples."""
    data = load_json(json_path)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError("Unrecognized LoCoMo JSON shape. Expect a list or {'samples': [...]}.")


def extract_sessions(conv_obj: Dict[str, Any]) -> List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]]:
    """
    Extract sessions as (idx, timestamp, turns, optional_session_summary).
    sessions are under keys like 'session_1', 'session_2', ... with paired 'session_1_date_time', 'session_1_summary' etc.
    """
    sessions: List[Tuple[int, str, List[Dict[str, Any]], Optional[str]]] = []
    for k, v in conv_obj.items():
        m = re.match(r'^session_(\d+)$', k)
        if not (m and isinstance(v, list)):
            continue
        idx = int(m.group(1))
        ts = conv_obj.get(f"session_{idx}_date_time", "")
        ssum = conv_obj.get(f"session_{idx}_summary", None)
        sessions.append((idx, ts, v, ssum if isinstance(ssum, str) and ssum.strip() else None))
    sessions.sort(key=lambda x: x[0])
    return sessions


def session_to_text(idx: int, ts: str, turns: List[Dict[str, Any]], session_summary: Optional[str]) -> str:
    lines = [f"[Session {idx} | {ts}]".strip()]
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        dia_id  = turn.get("dia_id", "")
        text    = turn.get("text", "")
        lines.append(f"{speaker} ({dia_id}): {text}")
    if session_summary:
        lines.append("")
        lines.append(f"Session {idx} summary: {session_summary}")
    return "\n".join(lines).strip()


def build_session_chunks_for_sample(sample: Dict[str, Any]) -> List[str]:
    conv = sample.get("conversation", {})
    sessions = extract_sessions(conv)
    chunks: List[str] = []
    for idx, ts, turns, ssum in sessions:
        chunks.append(session_to_text(idx, ts, turns, ssum))
    return chunks


def collect_qa_items_for_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    qas: List[Dict[str, Any]] = []
    sid = sample.get("sample_id", None)
    for q in sample.get("qa", []):
        qas.append({
            "sample_id": sid,
            "question": q.get("question"),
            "answer": q.get("answer"),
            "category": q.get("category"),
            "evidence": q.get("evidence"),
        })
    return qas


# ========== Prompt（保持原有风格） ==========
def make_memory_only_prompt(memory_obj: Any, question: str) -> str:
    mem_str = json.dumps(memory_obj, ensure_ascii=False, indent=2) if isinstance(memory_obj, dict) else str(memory_obj)
    return f"""
    Based on the MEMORY STATE below,  write an answer in the form of a brief short phrase for the following question. Answer with exact words from the context whenever possible.
    The date should be written as an exact date.

    MEMORY STATE:
    {mem_str}

    QUESTION:
    {question}

    Short answer:
    """


def make_memory_only_prompt_category3(memory_obj: Any, question: str) -> str:
    mem_str = json.dumps(memory_obj, ensure_ascii=False, indent=2) if isinstance(memory_obj, dict) else str(memory_obj)
    return f"""
    Based on the MEMORY STATE below,  write an answer in the form of a brief short phrase for the following question. Answer with exact words from the context whenever possible.
    The date should be written as an exact date.

    MEMORY STATE:
    {mem_str}

    QUESTION:
    {question}

    Short answer:
    """




def make_summary_prompt(summary: str, question: str) -> str:
    return f"""
    Based on the summary below, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.
    For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.
    If the question is about the duration, answer in the form of several years, months, or days.
    
    QUESTION:
    {question}

    SUMMARY:
    {summary}

    Short answer:
    """


def make_summary_prompt_category3(summary: str, question: str) -> str:
    return f"""
    Based on the summary below, write an answer in the form of a short phrase for the following question.
    The question may need you to analyze and infer the answer from the summary.
    
    QUESTION:
    {question}

    SUMMARY:
    {summary}

    Short answer:
    """


# ========== Memory & DeepResearch 封装 ==========
def build_memory_for_sample(llm, session_chunks: List[str]):
    """
    顺序运行 MemoryAgent：只负责把 sessions -> events state + abstracts。
    返回：(mem_agent, memory_history, final_memory_state, session_abstracts)
    """
    mem = MemoryAgent(llm)
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
                              max_sessions: int = 6) -> Dict[str, Any]:
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
    ret = DeepResearchAgent(qs_llm)
    result = ret.deep_research(question, final_memory, pages, max_sessions)
    
    return result


def deep_research_answer(qs_llm, final_memory, session_chunks: List[str],
                         question: str, session_abstracts: List[str] = None,
                         max_sessions: int = 6) -> Tuple[str, List[int]]:
    """
    用 DeepResearchAgent 取回相关 sessions + 总结。
    返回：(summary_text, used_session_ids)
    """
    result = memory_deep_research(
        qs_llm, final_memory, session_chunks, question, 
        session_abstracts=session_abstracts,
        max_sessions=max_sessions
    )
    session_ids = result.get("session_ids_used") or result.get("session_ids") or []
    summary = result.get("summary", "")

    return summary, session_ids

def answer_with_summary(qs_llm, category: Optional[int], summary: str, question: str) -> Dict[str, Any]:
    if category == 3:
        prompt = make_summary_prompt_category3(summary, question)
    else:
        prompt = make_summary_prompt(summary, question)
    raw = qs_llm.generate(prompt)
    return raw


def answer_with_memory(qs_llm, category: Optional[int], final_memory: Dict[str, Any], question: str) -> Dict[str, Any]:
    if category == 3:
        prompt = make_memory_only_prompt_category3(final_memory, question)
    else:
        prompt = make_memory_only_prompt(final_memory, question)
    raw = qs_llm.generate(prompt)
    return raw




# ========== 已有记忆载入/保存 ==========
def load_existing_memory(sample_id: str, outdir: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]], Optional[List[str]]]:
    base = os.path.join(outdir, sample_id)
    hist = os.path.join(base, "memory_history.json")
    finl = os.path.join(base, "final_memory.json")
    abstracts = os.path.join(base, "session_abstracts.json")
    if not (os.path.exists(hist) and os.path.exists(finl)):
        return None, None, None
    try:
        with open(hist, "r", encoding="utf-8") as f:
            memory_history = json.load(f)
        with open(finl, "r", encoding="utf-8") as f:
            final_memory = json.load(f)
        session_abstracts = None
        if os.path.exists(abstracts):
            with open(abstracts, "r", encoding="utf-8") as f:
                session_abstracts = json.load(f)
        return memory_history, final_memory, session_abstracts
    except Exception as e:
        print(f"[WARN] load_existing_memory({sample_id}) failed: {e}")
        return None, None, None


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_existing_qa_results(sample_id: str, outdir: str, mode: str) -> List[Dict[str, Any]]:
    """加载已有的QA结果，用于断点保护"""
    base = os.path.join(outdir, sample_id)
    if mode == "answer_only":
        qa_file = os.path.join(base, "qa_results_answer.json")
    elif mode == "build_and_answer":
        qa_file = os.path.join(base, "qa_results_answer.json")
    else:
        return []
    
    if not os.path.exists(qa_file):
        return []
    
    try:
        with open(qa_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] load_existing_qa_results({sample_id}) failed: {e}")
        return []


def is_question_answered(question: str, existing_results: List[Dict[str, Any]]) -> bool:
    """检查问题是否已经被回答过"""
    for result in existing_results:
        if result.get("question") == question:
            return True
    return False


# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser(description="LoCoMo QA with Memory Agent Architecture (v3)")
    parser.add_argument("--data", type=str, default=r"D:\python\Streamingllm\datasets\locomo\locomo10.json", help="Path to dataset")
    parser.add_argument("--mode", type=str, default="answer_only", choices=["memory_only", "answer_only", "build_and_answer"], help="Processing mode")
    parser.add_argument("--outdir", type=str, default=r"D:\python\Streamingllm\memory_agent\results\v3_results", help="Output directory")
    parser.add_argument("--max-sessions", type=int, default=8, help="Max sessions retrieved per question")
    parser.add_argument("--sample-idx", type=int, default=-1, help="Run a single sample (0-based). -1 = run all")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate first N samples (0 = all)")
    parser.add_argument("--model-memory", type=str, default="gpt-4o-mini", help="Model for memory processing")
    parser.add_argument("--model-research", type=str, default="gpt-4o-mini", help="Model for research")
    parser.add_argument("--model-answer", type=str, default="gpt-4o-mini", help="Model for answering")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Max tokens per session chunk")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep between requests (seconds)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing results")
    args = parser.parse_args()

    # 准备 LLM （全部使用 OpenRouterModel）
    llm_for_memory = OpenRouterModel(model=args.model_memory)
    llm_for_research = OpenRouterModel(model=args.model_research)
    llm_for_answer = OpenRouterModel(model=args.model_answer)

    # 载入数据
    samples_all = load_locomo(args.data)
    
    # 限制样本数量
    if args.limit > 0:
        samples = samples_all[args.start_idx:args.start_idx + args.limit]
    elif args.start_idx > 0:
        samples = samples_all[args.start_idx:]
    else:
        samples = samples_all
    
    if not isinstance(samples, list) or not samples:
        print("[Error] No samples found in dataset.")
        return

    # 选择范围
    indices = [args.sample_idx] if args.sample_idx >= 0 else list(range(len(samples)))

    for idx in indices:
        if idx < 0 or idx >= len(samples):
            print(f"[Warn] sample-idx {idx} out of range, skip.")
            continue

        sample = samples[idx]
        sample_id = sample.get("sample_id") or f"sample-{idx}"
        print(f"\n== {args.mode} | sample#{idx} ({sample_id}) ==")

        # 构建顺序的会话 list[str]（第 1 条即 session_id=1）
        session_chunks = build_session_chunks_for_sample(sample)
        print(f"  sessions: {len(session_chunks)}")

        # 输出目录
        sample_dir = os.path.join(args.outdir, sample_id)
        ensure_dir(sample_dir)

        if args.mode == "memory_only":
            mem_agent, history, final_memory, session_abstracts = build_memory_for_sample(llm_for_memory, session_chunks)
            save_json(history, os.path.join(sample_dir, "memory_history.json"))
            save_json(final_memory, os.path.join(sample_dir, "final_memory.json"))
            save_json(session_abstracts, os.path.join(sample_dir, "session_abstracts.json"))
            print("  [memory-only] memory saved.")

        elif args.mode == "answer_only":
            # 载入已有记忆
            memory_history, final_memory, session_abstracts = load_existing_memory(sample_id, args.outdir)
            
            # 将final_memory转换为字符串格式，便于后续作为prompt传入
            if final_memory is not None:
                final_memory_str = json.dumps(final_memory, ensure_ascii=False, indent=2)
                print(f"  [info] final_memory converted to string, length: {len(final_memory_str)} chars")
            if final_memory is None:
                print("  [answer-only] no final_memory found; run memory-only or build-and-answer first.")
                continue

            qas = collect_qa_items_for_sample(sample)
            if max_questions_demo is not None:
                qas = qas[:max_questions_demo]

            # 加载已有的QA结果用于断点保护
            qa_results = load_existing_qa_results(sample_id, args.outdir, args.mode)
            results_file = os.path.join(sample_dir, "qa_results_answer.json")
            
            print(f"  [info] 加载了 {len(qa_results)} 个已有的QA结果")
            
            for qi in tqdm(qas, desc="QA", unit="q"):
                q = qi.get("question") or ""
                gold = qi.get("answer")
                cat = qi.get("category")
                evd = qi.get("evidence")

                # 断点保护：检查问题是否已经被回答过
                if is_question_answered(q, qa_results):
                    print(f"  [skip] 问题已经被回答过: {q[:50]}...")
                    continue

                print(f"  [processing] 正在回答问题: {q[:50]}...")


                if cat == 5:
                    print(f"  [skip] 跳过category 5的问题: {q[:50]}...")
                    continue


                # DeepResearch（召回 + 总结）
                summary, session_ids = deep_research_answer(
                    llm_for_research, final_memory_str, session_chunks, q, 
                    session_abstracts=session_abstracts, max_sessions=args.max_sessions
                )

                # 两种短答（保持原 prompt 风格）
                summary_answer   = answer_with_summary(llm_for_answer, cat, summary, q)
                memory_answer = answer_with_memory(llm_for_answer, cat, final_memory_str, q)

                # 添加新的结果
                new_result = {
                    "question": q,
                    "gold_answer": gold,
                    "category": cat,
                    "evidence": evd,
                    "memory_answer": memory_answer,
                    "summary": summary,
                    "summary_answer": summary_answer,
                    "retrieved_session_ids": session_ids
                }
                qa_results.append(new_result)

                # 立即保存结果（增量保存）
                save_json(qa_results, results_file)
                print(f"  [saved] 已保存 {len(qa_results)} 个QA结果")

            print("  [answer-only] 所有结果已保存完成。")

        elif args.mode == "build_and_answer":
            # 先构建记忆
            mem_agent, history, final_memory, session_abstracts = build_memory_for_sample(llm_for_memory, session_chunks)
            save_json(history, os.path.join(sample_dir, "memory_history.json"))
            save_json(final_memory, os.path.join(sample_dir, "final_memory.json"))
            save_json(session_abstracts, os.path.join(sample_dir, "session_abstracts.json"))

            if final_memory is not None:
                final_memory_str = json.dumps(final_memory, ensure_ascii=False, indent=2)
                print(f"  [info] final_memory converted to string, length: {len(final_memory_str)} chars")

            # 再做 QA
            qas = collect_qa_items_for_sample(sample)
            if max_questions_demo is not None:
                qas = qas[:max_questions_demo]

            # 加载已有的QA结果用于断点保护
            qa_results = load_existing_qa_results(sample_id, args.outdir, args.mode)
            results_file = os.path.join(sample_dir, "qa_results_answer.json")
            
            print(f"  [info] 加载了 {len(qa_results)} 个已有的QA结果")
            
            for qi in tqdm(qas, desc="QA", unit="q"):
                q = qi.get("question") or ""
                gold = qi.get("answer")
                cat = qi.get("category")
                evd = qi.get("evidence")

                # 断点保护：检查问题是否已经被回答过
                if is_question_answered(q, qa_results):
                    print(f"  [skip] 问题已经被回答过: {q[:50]}...")
                    continue

                print(f"  [processing] 正在回答问题: {q[:50]}...")

                summary, session_ids = deep_research_answer(
                    llm_for_research, final_memory_str, session_chunks, q, 
                    session_abstracts=session_abstracts, max_sessions=args.max_sessions
                )

                summary_answer   = answer_with_summary(llm_for_answer, cat, summary, q)
                memory_answer = answer_with_memory(llm_for_answer, cat, final_memory_str, q)

                # 添加新的结果
                new_result = {
                    "question": q,
                    "gold_answer": gold,
                    "category": cat,
                    "evidence": evd,
                    "memory_answer": memory_answer,
                    "summary": summary,
                    "summary_answer": summary_answer,
                    "retrieved_session_ids": session_ids
                }
                qa_results.append(new_result)

                # 立即保存结果（增量保存）
                save_json(qa_results, results_file)
                print(f"  [saved] 已保存 {len(qa_results)} 个QA结果")

            print("  [build-and-answer] 所有结果已保存完成。")

        else:
            print(f"[Error] unknown mode: {args.mode}")
            return


if __name__ == "__main__":
    main()
