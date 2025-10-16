import os
import re
import json
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# === Project-local imports (adapt to your repo layout) ===
from agents_new import MemoryAgent, RetrievalAgent
from llm_call import HFModel, OpenRouterModel


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

# 并行处理控制
num_workers = 3  # 并行进程数，1表示串行处理，>1表示并行处理


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
        session_text = f"[Paragraph {i} | {title}]\n"
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
    """
    mem = MemoryAgent(llm)
    history = mem.run_memory_agent(sessions=session_chunks)
    final_state_raw = mem.get_final_memory_output()
    final_state = safe_json_extract(final_state_raw) or final_state_raw
    return mem, history, final_state


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


def retrieve_and_summary(llm, mem_agent, final_memory, session_chunks: List[str], question: str, max_sessions: int = 5) -> Tuple[Dict[str, Any], List[int], List[Dict[str, Any]]]:
    ret = RetrievalAgent(llm, mem_agent)

    # get session ids from retriever (LOCAL to this sample)
    try:
        result = ret.find_relevant_sessions(
            query=question,
            memory_output=final_memory,
            max_sessions=max_sessions,
        )
        session_ids = result.get("session_ids", [])
        summary = result.get("summary", "")
    except Exception as e:
        print("[WARN] RetrievalAgent failed:", e)
        session_ids = []
        summary = ""

    # fallback: last few sessions
    if not session_ids and len(session_chunks) > 0:
        session_ids = list(range(max(0, len(session_chunks) - 3), len(session_chunks)))

    # prefer MemoryAgent helper; fallback to direct mapping
    retrieved_struct: List[Dict[str, Any]] = []
    try:
        retrieved_struct = mem_agent.get_session_texts(session_ids, keep_order=True, unique=True)
    except Exception:
        for sid in session_ids:
            if isinstance(sid, int) and 0 <= sid < len(session_chunks):
                retrieved_struct.append({"session_id": sid, "session_text": session_chunks[sid]})

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


def split_samples_into_chunks(samples: List[Dict[str, Any]], num_workers: int) -> List[List[Dict[str, Any]]]:
    """将样本列表分割成多个块用于并行处理"""
    if num_workers <= 1:
        return [samples]
    
    chunk_size = max(1, len(samples) // num_workers)
    chunks = []
    
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks


def process_sample_chunk(args: Tuple) -> Tuple[int, str, str, str]:
    """
    处理一个样本块的函数，用于并行执行，支持增量保存
    返回: (chunk_id, memory_history_file, final_memory_file, qa_results_file)
    """
    chunk_id, samples_chunk, openrouter_model, qs_model, max_chunks_to_retrieve, results_dir = args
    
    print(f"[Worker {chunk_id}] 开始处理 {len(samples_chunk)} 个样本")
    
    # 初始化LLM
    llm = OpenRouterModel(model=openrouter_model)
    qs_llm = OpenRouterModel(model=qs_model)
    
    # 为每个进程创建独立的结果文件
    chunk_results_dir = os.path.join(results_dir, f"chunk_{chunk_id}")
    os.makedirs(chunk_results_dir, exist_ok=True)
    
    memory_history_file = os.path.join(chunk_results_dir, "memory_history.json")
    final_memory_file = os.path.join(chunk_results_dir, "final_memory.json")
    qa_results_file = os.path.join(chunk_results_dir, "qa_results.json")
    
    # 加载已有的结果（支持断点续传）
    memory_history_chunk, final_memory_chunk, qa_results_chunk = load_existing_results(
        memory_history_file, final_memory_file, qa_results_file
    )
    
    chunk_f1_memory = 0.0
    chunk_f1_retrieval = 0.0
    chunk_f1_summary = 0.0
    chunk_questions = 0
    
    for si, sample in enumerate(samples_chunk):
        sample_id = sample.get("_id", f"sample_{si}")
        
        # 检查是否已经处理过这个样本
        if sample_id in qa_results_chunk:
            print(f"[Worker {chunk_id}] 样本 {sample_id} 已存在，跳过...")
            # 更新统计信息
            sample_data = qa_results_chunk[sample_id]
            sample_stats = sample_data.get("sample_stats", {})
            num_questions = sample_stats.get("num_questions", 0)
            avg_f1_memory = sample_stats.get("avg_f1_memory", 0.0)
            avg_f1_retrieval = sample_stats.get("avg_f1_retrieval", 0.0)
            avg_f1_summary = sample_stats.get("avg_f1_summary", 0.0)
            
            chunk_f1_memory += avg_f1_memory * num_questions
            chunk_f1_retrieval += avg_f1_retrieval * num_questions
            chunk_f1_summary += avg_f1_summary * num_questions
            chunk_questions += num_questions
            continue
        
        print(f"[Worker {chunk_id}] 处理样本 {si+1}/{len(samples_chunk)}: {sample_id}")
        
        try:
            # 构建session chunks
            session_chunks = build_context_sessions(sample["context"])
            
            # 构建记忆
            mem_agent, history, final_memory = build_memory_for_sample(llm, session_chunks)
            
            # 收集QA项目
            qas = collect_qa_items_for_sample(sample)
            
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
                
                # 三种方法回答问题
                a_memory = answer_memory_only(qs_llm, final_memory, q)
                summary, session_ids, retrieved = retrieve_and_summary(
                    qs_llm, mem_agent, final_memory, session_chunks, q, 
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
            
            # 将样本结果添加到对应的字典中
            memory_history_chunk[sample_id] = history
            final_memory_chunk[sample_id] = final_memory
            qa_results_chunk[sample_id] = {
                "sample_stats": sample_stats,
                "qa_results": qa_results,
                "num_questions": len(qas)
            }
            
            # 更新块统计
            chunk_f1_memory += sample_f1_memory
            chunk_f1_retrieval += sample_f1_retrieval
            chunk_f1_summary += sample_f1_summary
            chunk_questions += len(qas)
            
            # 增量保存：每处理完一个样本就保存一次
            save_results(memory_history_file, final_memory_file, qa_results_file,
                       memory_history_chunk, final_memory_chunk, qa_results_chunk)
            
            print(f"[Worker {chunk_id}] 样本 {sample_id} 完成，F1(Memory)={sample_f1_memory/denom:.3f}, F1(Retrieval)={sample_f1_retrieval/denom:.3f}, F1(Summary)={sample_f1_summary/denom:.3f}")
            
        except Exception as e:
            print(f"[Worker {chunk_id}] 处理样本 {sample_id} 时出错: {e}")
            # 即使出错也要保存已处理的结果
            save_results(memory_history_file, final_memory_file, qa_results_file,
                       memory_history_chunk, final_memory_chunk, qa_results_chunk)
            continue
    
    print(f"[Worker {chunk_id}] 完成处理，共处理 {len(samples_chunk)} 个样本，{chunk_questions} 个问题")
    print(f"[Worker {chunk_id}] 结果已保存到: {chunk_results_dir}")
    
    return chunk_id, memory_history_file, final_memory_file, qa_results_file


def merge_chunk_results(chunk_files: List[Tuple[int, str, str, str]], 
                       memory_history_file: str, final_memory_file: str, qa_results_file: str) -> None:
    """
    合并多个进程的结果文件，保留已有的结果
    chunk_files: [(chunk_id, memory_history_file, final_memory_file, qa_results_file), ...]
    """
    print("\n开始合并各进程的结果文件...")
    
    # 先加载已有的结果文件
    existing_memory_history, existing_final_memory, existing_qa_results = load_existing_results(
        memory_history_file, final_memory_file, qa_results_file
    )
    
    merged_memory_history = existing_memory_history.copy()
    merged_final_memory = existing_final_memory.copy()
    merged_qa_results = existing_qa_results.copy()
    
    for chunk_id, mem_file, final_file, qa_file in chunk_files:
        print(f"  合并块 {chunk_id} 的结果...")
        
        try:
            # 读取memory_history
            if os.path.exists(mem_file):
                with open(mem_file, 'r', encoding='utf-8') as f:
                    chunk_memory_history = json.load(f)
                merged_memory_history.update(chunk_memory_history)
            
            # 读取final_memory
            if os.path.exists(final_file):
                with open(final_file, 'r', encoding='utf-8') as f:
                    chunk_final_memory = json.load(f)
                merged_final_memory.update(chunk_final_memory)
            
            # 读取qa_results
            if os.path.exists(qa_file):
                with open(qa_file, 'r', encoding='utf-8') as f:
                    chunk_qa_results = json.load(f)
                merged_qa_results.update(chunk_qa_results)
                
        except Exception as e:
            print(f"  警告: 合并块 {chunk_id} 时出错: {e}")
            continue
    
    # 保存合并后的结果
    save_results(memory_history_file, final_memory_file, qa_results_file,
               merged_memory_history, merged_final_memory, merged_qa_results)
    
    print(f"  合并完成，共合并 {len(merged_memory_history)} 个样本的结果")
    
    # 清理临时文件
    for chunk_id, mem_file, final_file, qa_file in chunk_files:
        try:
            chunk_dir = os.path.dirname(mem_file)
            if os.path.exists(chunk_dir):
                import shutil
                shutil.rmtree(chunk_dir)
                print(f"  已清理临时目录: {chunk_dir}")
        except Exception as e:
            print(f"  警告: 清理临时目录时出错: {e}")


def run_serial_processing(samples: List[Dict[str, Any]], openrouter_model: str, qs_model: str, 
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
        mem_agent, history, final_memory = build_memory_for_sample(llm, session_chunks)
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
                qs_llm, mem_agent, final_memory, session_chunks, q, 
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


def find_existing_chunk_dirs(results_dir: str) -> List[str]:
    """查找已存在的分片目录"""
    chunk_dirs = []
    if not os.path.exists(results_dir):
        return chunk_dirs
    
    for item in os.listdir(results_dir):
        if item.startswith("chunk_"):
            chunk_path = os.path.join(results_dir, item)
            if os.path.isdir(chunk_path):
                chunk_dirs.append(chunk_path)
    
    return sorted(chunk_dirs)


def collect_completed_samples_from_chunks(chunk_dirs: List[str]) -> set:
    """从所有分片目录中收集已完成的样本ID"""
    completed_samples = set()
    
    for chunk_dir in chunk_dirs:
        qa_results_file = os.path.join(chunk_dir, "qa_results.json")
        if os.path.exists(qa_results_file):
            try:
                with open(qa_results_file, 'r', encoding='utf-8') as f:
                    chunk_qa_results = json.load(f)
                
                for sample_id, sample_data in chunk_qa_results.items():
                    if sample_id == "global_stats":
                        continue
                    if check_sample_completed(sample_id, {sample_id: sample_data}):
                        completed_samples.add(sample_id)
                        
            except Exception as e:
                print(f"  警告: 读取分片 {chunk_dir} 的QA结果时出错: {e}")
    
    return completed_samples


def merge_existing_chunks(results_dir: str, memory_history_file: str, final_memory_file: str, qa_results_file: str) -> None:
    """合并已存在的分片结果到主结果文件"""
    chunk_dirs = find_existing_chunk_dirs(results_dir)
    if not chunk_dirs:
        return
    
    print(f"  发现 {len(chunk_dirs)} 个已存在的分片目录，开始合并...")
    
    chunk_files = []
    for chunk_dir in chunk_dirs:
        chunk_id = os.path.basename(chunk_dir).replace("chunk_", "")
        memory_history_file_chunk = os.path.join(chunk_dir, "memory_history.json")
        final_memory_file_chunk = os.path.join(chunk_dir, "final_memory.json")
        qa_results_file_chunk = os.path.join(chunk_dir, "qa_results.json")
        
        # 检查文件是否存在
        if (os.path.exists(memory_history_file_chunk) and 
            os.path.exists(final_memory_file_chunk) and 
            os.path.exists(qa_results_file_chunk)):
            chunk_files.append((int(chunk_id), memory_history_file_chunk, final_memory_file_chunk, qa_results_file_chunk))
    
    if chunk_files:
        merge_chunk_results(chunk_files, memory_history_file, final_memory_file, qa_results_file)
        print(f"  已合并 {len(chunk_files)} 个分片的结果")


def run_parallel_processing(samples: List[Dict[str, Any]], num_workers: int, openrouter_model: str, 
                           qs_model: str, max_chunks_to_retrieve: int, results_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """并行处理样本，支持断点续传"""
    print(f"\nSTEP 2) 准备并行处理，使用 {num_workers} 个进程...")
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 定义三个独立的结果文件
    memory_history_file = os.path.join(results_dir, "memory_history.json")
    final_memory_file = os.path.join(results_dir, "final_memory.json")
    qa_results_file = os.path.join(results_dir, "qa_results.json")
    
    # 首先合并已存在的分片结果
    merge_existing_chunks(results_dir, memory_history_file, final_memory_file, qa_results_file)
    
    # 加载现有结果（包括刚合并的分片结果）
    memory_history_all, final_memory_all, qa_results_all = load_existing_results(
        memory_history_file, final_memory_file, qa_results_file
    )
    
    # 收集所有分片目录中已完成的样本
    chunk_dirs = find_existing_chunk_dirs(results_dir)
    completed_samples_from_chunks = collect_completed_samples_from_chunks(chunk_dirs)
    
    # 过滤掉已完成的样本（包括主结果文件和分片中的）
    samples_to_process = []
    for sample in samples:
        sample_id = sample.get("_id", "")
        if (not check_sample_completed(sample_id, qa_results_all) and 
            sample_id not in completed_samples_from_chunks):
            samples_to_process.append(sample)
    
    if not samples_to_process:
        print("  所有样本都已完成处理，无需重新处理")
        return memory_history_all, final_memory_all, qa_results_all
    
    print(f"  需要处理的样本数: {len(samples_to_process)}")
    print(f"  从分片中发现已完成的样本数: {len(completed_samples_from_chunks)}")
    
    # 分割样本为多个块
    sample_chunks = split_samples_into_chunks(samples_to_process, num_workers)
    print(f"  样本分片数: {len(sample_chunks)}")
    
    # 准备并行任务参数
    task_args = []
    for i, chunk in enumerate(sample_chunks):
        task_args.append((i, chunk, openrouter_model, qs_model, max_chunks_to_retrieve, results_dir))
    
    print(f"\nSTEP 3) 开始并行处理...")
    start_time = time.time()
    
    # 使用进程池并行处理
    chunk_files = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_chunk = {executor.submit(process_sample_chunk, args): args[0] for args in task_args}
        
        # 收集结果文件路径
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                chunk_id, memory_history_file_chunk, final_memory_file_chunk, qa_results_file_chunk = future.result()
                chunk_files.append((chunk_id, memory_history_file_chunk, final_memory_file_chunk, qa_results_file_chunk))
                print(f"[主进程] 收到块 {chunk_id} 的结果文件")
                
            except Exception as e:
                print(f"[主进程] 处理块 {chunk_id} 时出错: {e}")
    
    end_time = time.time()
    print(f"  并行处理完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 合并所有进程的结果文件
    if chunk_files:
        merge_chunk_results(chunk_files, memory_history_file, final_memory_file, qa_results_file)
        
        # 重新加载合并后的结果
        memory_history_all, final_memory_all, qa_results_all = load_existing_results(
            memory_history_file, final_memory_file, qa_results_file
        )
    
    return memory_history_all, final_memory_all, qa_results_all


# =====================
# Main execution
# =====================
if __name__ == "__main__":
    print("STEP 1) Load HotpotQA:", hotpotqa_json)
    samples = load_hotpotqa(hotpotqa_json)

    # 限制样本数量
    if max_samples is not None:
        samples = samples[:max_samples]
    
    print(f"  samples: {len(samples)}")
    print(f"  并行进程数: {num_workers}")

    # 根据并行进程数选择处理方式
    if num_workers <= 1:
        # 串行处理
        memory_history_all, final_memory_all, qa_results_all = run_serial_processing(
            samples, openrouter_model, qs_model, max_chunks_to_retrieve, results_dir
        )
    else:
        # 并行处理
        memory_history_all, final_memory_all, qa_results_all = run_parallel_processing(
            samples, num_workers, openrouter_model, qs_model, max_chunks_to_retrieve, results_dir
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
