import os
import re
import json
import argparse
import time
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

# === Project-local imports (adapt to your repo layout) ===
from gam.agents import (
    MemoryAgent, 
    DeepResearchAgent
)
from gam.utils import (
    build_session_chunks_from_text,
    build_pages_from_sessions_and_abstracts
)
from gam.llm_call import HFModel, OpenRouterModel


# =====================
# CONFIG — EDIT HERE
# =====================

results_dir = r"D:\python\Streamingllm\memory_agent\hotpotqa_results"

# Path to HotpotQA JSON
hotpotqa_json = r"D:\python\Streamingllm\datasets\hotpotqa\hotpot_dev_distractor_v1.json"

# Choose model backend
openrouter_model = "openai/gpt-4o-mini"
qs_model = "openai/gpt-4o-mini"

# Retrieval parameters
max_chunks_to_retrieve = 5

# Quick eval parameters
max_questions_demo = None  # set to None for all

# 运行控制
max_samples = None  # 限制处理的样本数量，None表示处理所有样本


# =====================
# Utils
# =====================

def safe_json_extract(candidate: Any) -> Optional[Dict[str, Any]]:
    """Try to parse a model's output (string or dict) into dict. Return None if fail."""
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


def normalize_text(s: str) -> str:
    """Lowercase, strip, compress whitespace, remove surrounding quotes/punctuation."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'^[\"\'\`\(\)\[\]\{\}\.\,\:\;\-_\s]+', '', s)
    s = re.sub(r'[\"\'\`\(\)\[\]\{\}\.\,\:\;\-_\s]+$', '', s)
    return s


def f1_score(prediction: str, gold: str) -> float:
    """计算F1分数"""
    p = normalize_text(prediction).split()
    g = normalize_text(gold).split()
    if not p and not g: 
        return 1.0
    if not p or not g:  
        return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:   
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


# =====================
# HotpotQA loading & processing
# =====================

def load_hotpotqa(json_path: str) -> List[Dict[str, Any]]:
    """Load HotpotQA JSON and return the list of samples."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_context_sessions(context: List[List[Any]]) -> List[str]:
    """
    将HotpotQA的context转换为session格式
    context: [[title, [sent1, sent2, ...]], ...]
    返回: List[str] 每个元素是一个session的文本
    """
    sessions = []
    for i, (title, sentences) in enumerate(context, 1):
        # 将每个段落作为一个session
        session_text = f"[Session {i} | {title}]\n"
        session_text += " ".join(sentences)
        sessions.append(session_text)
    return sessions


def collect_qa_items_for_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从HotpotQA样本中提取QA信息"""
    qas = []
    sample_id = sample.get("_id", "")
    
    qas.append({
        "sample_id": sample_id,
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "type": sample.get("type", "comparison"),  # HotpotQA的问题类型
        "level": sample.get("level", "medium"),    # 难度级别
        "supporting_facts": sample.get("supporting_facts", []),
    })
    return qas


# =====================
# Prompt builders
# =====================

def make_memory_only_prompt(memory_obj: Any, question: str) -> str:
    mem_str = json.dumps(memory_obj, ensure_ascii=False, indent=2) if isinstance(memory_obj, dict) else str(memory_obj)
    return f"""
You are a careful multi-hop reading assistant. 
Use the given MEMORY below, 
Answer with ONLY the final answer string; no extra words.

MEMORY:
{mem_str}

QUESTION:
{question}

Answer:
"""


def make_retrieval_prompt(retrieved: List[Dict[str, Any]], question: str) -> str:
    ctx = []
    for it in retrieved:
        para_id = it.get("session_id")
        para_text = it.get("session_text", "")
        ctx.append(f"[Paragraph {para_id}]\n{para_text}")
    ctx_str = "\n\n---\n\n".join(ctx)
    return f"""
You are a careful multi-hop reading assistant. 
Use the given PARAGRAPHS below, 
Answer with ONLY the final answer string; no extra words.

PARAGRAPHS:
{ctx_str}

QUESTION:
{question}

Answer:
"""


def make_summary_prompt(summary: str, question: str) -> str:
    return f"""
You are a careful multi-hop reading assistant. 
Use the given SUMMARY below, 
Answer with ONLY the final answer string; no extra words.

QUESTION:
{question}

SUMMARY:
{summary}

Answer:
"""


# =====================
# Core: memory & QA
# =====================

def build_memory_for_sample(llm, session_chunks: List[str]):
    """
    Run MemoryAgent over the sessions (chunks) of ONE sample.
    session ids are LOCAL to this sample's MemoryAgent run.
    返回：(mem_agent, memory_history, final_memory_state, session_abstracts)
    """
    mem = MemoryAgent(llm)
    history = mem.run_memory_agent(sessions=session_chunks)
    
    # 使用新的get_memory_with_abstracts方法
    memory_with_abstracts = mem.get_memory_with_abstracts()
    final_state = memory_with_abstracts
    session_abstracts = memory_with_abstracts.get('session_abstracts', [])
    
    return mem, history, final_state, session_abstracts


def load_existing_memory(sample_id: str, results_dir: str):
    """
    Load existing memory files for a sample.
    Returns (memory_history, final_memory) or (None, None) if files don't exist.
    """
    sample_results_dir = os.path.join(results_dir, sample_id)
    history_file = os.path.join(sample_results_dir, "memory_history.json")
    final_memory_file = os.path.join(sample_results_dir, "final_memory.json")
    
    if not os.path.exists(history_file) or not os.path.exists(final_memory_file):
        return None, None
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            memory_history = json.load(f)
        with open(final_memory_file, 'r', encoding='utf-8') as f:
            final_memory = json.load(f)
        return memory_history, final_memory
    except Exception as e:
        print(f"  Error loading memory files for {sample_id}: {e}")
        return None, None


def answer_memory_only(llm, final_memory, question: str) -> Dict[str, Any]:
    prompt = make_memory_only_prompt(final_memory, question)
    raw = llm.generate(prompt)
    parsed = safe_json_extract(raw) or {"answer": str(raw)}
    return parsed


def retrieve_and_summary(llm, final_memory, session_chunks: List[str], question: str, 
                        session_abstracts: List[str] = None, max_sessions: int = 5) -> Tuple[str, List[int], List[Dict[str, Any]]]:
    """
    使用DeepResearchAgent进行检索和总结
    """
    # 构建pages
    if session_abstracts:
        pages = build_pages_from_sessions_and_abstracts(session_chunks, session_abstracts)
    else:
        # 如果没有abstracts，直接使用sessions作为pages
        pages = session_chunks
    
    # 使用DeepResearchAgent进行research
    ret = DeepResearchAgent(llm)
    result = ret.deep_research(question, final_memory, pages, max_sessions)
    
    session_ids = result.get("session_ids_used", [])
    summary = result.get("summary", "")
    
    # 构建retrieved_struct用于兼容性
    retrieved_struct = []
    for sid in session_ids:
        if isinstance(sid, int) and 1 <= sid <= len(session_chunks):
            retrieved_struct.append({
                "session_id": sid, 
                "session_text": session_chunks[sid-1]  # 转换为0-based索引
            })

    return summary, session_ids, retrieved_struct


def answer_with_summary(llm, summary, question: str) -> Dict[str, Any]:
    prompt = make_summary_prompt(summary, question)
    raw = llm.generate(prompt)
    parsed = safe_json_extract(raw) or {"answer": str(raw)}
    return parsed


def answer_with_retrieval(llm, retrieved_struct, question: str) -> Dict[str, Any]:
    prompt = make_retrieval_prompt(retrieved_struct, question)
    raw = llm.generate(prompt)
    parsed = safe_json_extract(raw) or {"answer": str(raw)}
    return parsed


def load_existing_results(memory_history_file: str, final_memory_file: str, qa_results_file: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """加载现有的结果文件"""
    # 加载memory_history
    memory_history = {}
    if os.path.exists(memory_history_file):
        try:
            with open(memory_history_file, 'r', encoding='utf-8') as f:
                memory_history = json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load memory history: {e}")
    
    # 加载final_memory
    final_memory = {}
    if os.path.exists(final_memory_file):
        try:
            with open(final_memory_file, 'r', encoding='utf-8') as f:
                final_memory = json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load final memory: {e}")
    
    # 加载qa_results
    qa_results = {}
    if os.path.exists(qa_results_file):
        try:
            with open(qa_results_file, 'r', encoding='utf-8') as f:
                qa_results = json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load qa results: {e}")
    
    return memory_history, final_memory, qa_results


def save_results(memory_history_file: str, final_memory_file: str, qa_results_file: str, 
                memory_history: Dict[str, Any], final_memory: Dict[str, Any], qa_results: Dict[str, Any]):
    """保存结果到文件"""
    with open(memory_history_file, 'w', encoding='utf-8') as f:
        json.dump(memory_history, f, ensure_ascii=False, indent=2)
    
    with open(final_memory_file, 'w', encoding='utf-8') as f:
        json.dump(final_memory, f, ensure_ascii=False, indent=2)
    
    with open(qa_results_file, 'w', encoding='utf-8') as f:
        json.dump(qa_results, f, ensure_ascii=False, indent=2)


def check_sample_completed(sample_id: str, qa_results: Dict[str, Any]) -> bool:
    """检查样本是否已完成处理"""
    if sample_id not in qa_results:
        return False
    
    sample_qa_data = qa_results[sample_id]
    qa_list = sample_qa_data.get("qa_results", [])
    num_questions = sample_qa_data.get("num_questions", 0)
    
    return len(qa_list) >= num_questions and num_questions > 0


def run_processing(samples: List[Dict[str, Any]], openrouter_model: str, qs_model: str, 
                  max_chunks_to_retrieve: int, results_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """串行处理样本"""
    print("\nSTEP 2) Prepare LLM ...")
    llm = OpenRouterModel(model=openrouter_model)
    qs_llm = OpenRouterModel(model=qs_model)
    print("  LLM ready.")

    print("\nSTEP 3) Processing all samples (串行模式)...")
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 定义三个独立的结果文件
    memory_history_file = os.path.join(results_dir, "memory_history.json")
    final_memory_file = os.path.join(results_dir, "final_memory.json")
    qa_results_file = os.path.join(results_dir, "qa_results.json")
    
    # 加载现有结果
    memory_history_all, final_memory_all, qa_results_all = load_existing_results(
        memory_history_file, final_memory_file, qa_results_file
    )
    
    total_f1_memory = 0.0
    total_f1_retrieval = 0.0
    total_f1_summary = 0.0
    total_questions = 0
    
    for si, sample in enumerate(tqdm(samples, desc="处理样本", unit="样本")):
        sample_id = sample.get("_id", f"sample_{si}")
        
        # 检查是否已完成
        if check_sample_completed(sample_id, qa_results_all):
            print(f"  Sample {si+1}/{len(samples)} ({sample_id}) already completed, skipping...")
            continue
        
        print(f"\n=== SAMPLE {si+1}/{len(samples)}: {sample_id} ===")
        
        # 构建session chunks
        session_chunks = build_context_sessions(sample["context"])
        print(f"  Sessions (chunks): {len(session_chunks)}")
        
        # 构建记忆
        mem_agent, history, final_memory, session_abstracts = build_memory_for_sample(llm, session_chunks)
        print(f"  Memory steps: {len(history)}")

        # 收集QA项目
        qas = collect_qa_items_for_sample(sample)
        print(f"  QA count: {len(qas)}")

        # 初始化QA结果列表
        qa_results = []
        sample_f1_memory = 0.0
        sample_f1_retrieval = 0.0
        sample_f1_summary = 0.0

        # 对每个问题进行评估
        for i, qi in enumerate(qas):
            q = qi["question"]
            gold = qi["answer"]
            q_type = qi["type"]
            level = qi["level"]

            print(f"    Processing QA {i+1}/{len(qas)}: {q[:50]}...")

            # 三种方法回答问题
            a_memory = answer_memory_only(qs_llm, final_memory, q)
            summary, session_ids, retrieved = retrieve_and_summary(
                qs_llm, final_memory, session_chunks, q, 
                session_abstracts=session_abstracts,
                max_sessions=max_chunks_to_retrieve
            )
            a_retrieval = answer_with_retrieval(qs_llm, retrieved, q)
            a_summary = answer_with_summary(qs_llm, summary, q)

            # 计算F1分数
            f1_memory = f1_score(a_memory.get("answer", ""), gold)
            f1_retrieval = f1_score(a_retrieval.get("answer", ""), gold)
            f1_summary = f1_score(a_summary.get("answer", ""), gold)

            sample_f1_memory += f1_memory
            sample_f1_retrieval += f1_retrieval
            sample_f1_summary += f1_summary

            # 构建当前问题的结果
            current_qa_result = {
                "q_idx": i,
                "question": q,
                "gold_answer": gold,
                "type": q_type,
                "level": level,
                "supporting_facts": qi["supporting_facts"],
                "memory_only": a_memory,
                "retrieval": a_retrieval,
                "summary": a_summary,
                "retrieved_session_ids": session_ids,
                "f1_memory": f1_memory,
                "f1_retrieval": f1_retrieval,
                "f1_summary": f1_summary,
            }
            qa_results.append(current_qa_result)
            
            print(f"      F1(Memory)={f1_memory:.3f} F1(Retrieval)={f1_retrieval:.3f} F1(Summary)={f1_summary:.3f}")

        # 计算样本统计
        denom = max(1, len(qas))
        sample_stats = {
            "sample_id": sample_id,
            "num_sessions": len(session_chunks),
            "num_questions": len(qas),
            "avg_f1_memory": sample_f1_memory / denom,
            "avg_f1_retrieval": sample_f1_retrieval / denom,
            "avg_f1_summary": sample_f1_summary / denom,
        }
        print(f"  >> Sample stats: {sample_stats}")
        
        # 将样本结果添加到对应的字典中
        memory_history_all[sample_id] = history
        final_memory_all[sample_id] = final_memory
        qa_results_all[sample_id] = {
            "sample_stats": sample_stats,
            "qa_results": qa_results,
            "num_questions": len(qas)
        }
        
        # 立即保存结果到三个文件
        save_results(memory_history_file, final_memory_file, qa_results_file,
                   memory_history_all, final_memory_all, qa_results_all)
        print(f"  Results saved to: {memory_history_file}, {final_memory_file}, {qa_results_file}")
        
        # 更新全局统计
        total_f1_memory += sample_f1_memory
        total_f1_retrieval += sample_f1_retrieval
        total_f1_summary += sample_f1_summary
        total_questions += len(qas)

    return memory_history_all, final_memory_all, qa_results_all




# =====================
# Main execution
# =====================
def main():
    parser = argparse.ArgumentParser(description="HotpotQA QA with Memory Agent Architecture (v3)")
    parser.add_argument("--data", type=str, default=hotpotqa_json, help="Path to dataset")
    parser.add_argument("--mode", type=str, default="build_and_answer", choices=["memory_only", "answer_only", "build_and_answer"], help="Processing mode")
    parser.add_argument("--outdir", type=str, default=results_dir, help="Output directory")
    parser.add_argument("--max-sessions", type=int, default=max_chunks_to_retrieve, help="Max sessions retrieved per question")
    parser.add_argument("--sample-idx", type=int, default=-1, help="Run a single sample (0-based). -1 = run all")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate first N samples (0 = all)")
    parser.add_argument("--model-memory", type=str, default=openrouter_model, help="Model for memory processing")
    parser.add_argument("--model-research", type=str, default=qs_model, help="Model for research")
    parser.add_argument("--model-answer", type=str, default=qs_model, help="Model for answering")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Max tokens per session chunk")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep between requests (seconds)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing results")
    
    args = parser.parse_args()

    print("STEP 1) Load HotpotQA:", args.data)
    samples = load_hotpotqa(args.data)

    # 限制样本数量
    if args.limit > 0:
        samples = samples[args.start_idx:args.start_idx + args.limit]
    elif args.start_idx > 0:
        samples = samples[args.start_idx:]
    
    print(f"  samples: {len(samples)}")

    # 串行处理
    memory_history_all, final_memory_all, qa_results_all = run_processing(
        samples, args.model_memory, args.model_answer, args.max_sessions, args.outdir
    )

    # 计算并保存全局统计
    total_questions = 0
    total_f1_memory = 0.0
    total_f1_retrieval = 0.0
    total_f1_summary = 0.0
    
    for sample_id, sample_data in qa_results_all.items():
        if sample_id == "global_stats":
            continue
        sample_stats = sample_data.get("sample_stats", {})
        num_questions = sample_stats.get("num_questions", 0)
        avg_f1_memory = sample_stats.get("avg_f1_memory", 0.0)
        avg_f1_retrieval = sample_stats.get("avg_f1_retrieval", 0.0)
        avg_f1_summary = sample_stats.get("avg_f1_summary", 0.0)
        
        total_questions += num_questions
        total_f1_memory += avg_f1_memory * num_questions
        total_f1_retrieval += avg_f1_retrieval * num_questions
        total_f1_summary += avg_f1_summary * num_questions
    
    if total_questions > 0:
        global_avg_f1_memory = total_f1_memory / total_questions
        global_avg_f1_retrieval = total_f1_retrieval / total_questions
        global_avg_f1_summary = total_f1_summary / total_questions
        
        global_stats = {
            "total_samples": len([k for k in qa_results_all.keys() if k != "global_stats"]),
            "total_questions": total_questions,
            "avg_f1_memory": global_avg_f1_memory,
            "avg_f1_retrieval": global_avg_f1_retrieval,
            "avg_f1_summary": global_avg_f1_summary,
        }
        
        # 将全局统计添加到qa_results文件中
        qa_results_all["global_stats"] = global_stats
        
        # 最终保存
        memory_history_file = os.path.join(results_dir, "memory_history.json")
        final_memory_file = os.path.join(results_dir, "final_memory.json")
        qa_results_file = os.path.join(results_dir, "qa_results.json")
        
        save_results(memory_history_file, final_memory_file, qa_results_file,
                   memory_history_all, final_memory_all, qa_results_all)
        
        print("\n== FINAL SUMMARY ==")
        print(f"  Total samples processed: {global_stats['total_samples']}")
        print(f"  Total questions: {total_questions}")
        print(f"  Global Average F1 (memory-only): {global_avg_f1_memory:.4f}")
        print(f"  Global Average F1 (retrieval): {global_avg_f1_retrieval:.4f}")
        print(f"  Global Average F1 (summary): {global_avg_f1_summary:.4f}")
        print(f"  Memory history saved to: {memory_history_file}")
        print(f"  Final memory saved to: {final_memory_file}")
        print(f"  QA results saved to: {qa_results_file}")


if __name__ == "__main__":
    main()
