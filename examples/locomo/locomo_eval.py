import json, re, math, csv, os
from collections import defaultdict, Counter

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)   # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)  # drop english articles
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str):
    s = normalize_text(s)
    return s.split() if s else []

def f1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def bleu1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if len(ptoks) == 0:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks) if ptoks else 0.0
    if ptoks and gtoks:
        bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks)/len(ptoks))
    else:
        bp = 0.0
    return bp * precision

def compute_metrics_by_category(items, pred_key: str = "retrieval", pred_field: str = "answer"):
    agg = defaultdict(list)
    rows = []
    for ex in items:
        cat = ex.get("category", "NA")
        gold = ex.get("gold_answer", "")
        pred = ""
        val = ex.get(pred_key, "")
        if isinstance(val, dict):
            pred = val.get(pred_field, "")
        else:
            pred = val
        f1 = f1_score(pred, gold)
        b1 = bleu1_score(pred, gold)
        agg[cat].append((f1, b1))
        rows.append({
            "q_idx": ex.get("q_idx", ""),
            "category": cat,
            "gold_answer": str(gold),
            "prediction": str(pred),
            "F1": f1,
            "BLEU1": b1
        })
    summary = []
    for cat in sorted(agg.keys(), key=lambda x: str(x)):
        scores = agg[cat]
        if scores:
            f1_avg = sum(s[0] for s in scores)/len(scores)
            b1_avg = sum(s[1] for s in scores)/len(scores)
            summary.append({"category": cat, "count": len(scores), "F1_avg": f1_avg, "BLEU1_avg": b1_avg})
    return summary, rows

def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    # === Edit these 4 lines to use your own paths/keys ===
    INPUT_DIR = r"D:\python\Streamingllm\memory_agent\results\v3_results"  # 改为文件夹路径
    OUTPUT_DIR = r"D:\python\Streamingllm\memory_agent\results\v3_results"  # 输出到同一文件夹
    PRED_KEYS = ["memory_answer", "summary_answer"]     # 评估两种答案类型
    PRED_FIELD = "answer"
    # =====================================================

    # 获取所有子文件夹
    subdirs = []
    for item in os.listdir(INPUT_DIR):
        item_path = os.path.join(INPUT_DIR, item)
        if os.path.isdir(item_path) and item.startswith('conv-'):
            subdirs.append(item)
    
    subdirs.sort()  # 按文件夹名称排序
    print(f"找到 {len(subdirs)} 个对话文件夹: {subdirs}")
    
    # 合并所有数据
    all_data = []
    for subdir in subdirs:
        input_json = os.path.join(INPUT_DIR, subdir, "qa_results_answer.json")  # 新格式文件名
        
        # 检查输入文件是否存在
        if not os.path.exists(input_json):
            print(f"跳过 {subdir}: 找不到 {input_json}")
            continue
            
        print(f"读取文件夹: {subdir}")
        
        try:
            with open(input_json, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                # 新格式：数据是直接的JSON数组
                if isinstance(data, list):
                    questions = data
                    # 为每个问题添加来源文件夹信息
                    for item in questions:
                        item['source_folder'] = subdir
                    all_data.extend(questions)
                    print(f"  添加了 {len(questions)} 条数据")
                else:
                    print(f"  警告: {subdir} 中数据格式不是数组")
                
        except Exception as e:
            print(f"读取 {subdir} 时出错: {e}")
            continue
    
    print(f"\n总共合并了 {len(all_data)} 条数据")
    
    # 对所有合并的数据进行整体分析
    for key in PRED_KEYS:
        print(f"\n# LoCoMo Metrics for pred_key='{key}', pred_field='{PRED_FIELD}' (所有数据)")
        summary, details = compute_metrics_by_category(all_data, pred_key=key, pred_field=PRED_FIELD)
        
        # 保存整体结果
        sum_csv = os.path.join(OUTPUT_DIR, f"locomo_metrics_{key}_all_data_summary.csv")
        det_csv = os.path.join(OUTPUT_DIR, f"locomo_metrics_{key}_all_data_details.csv")
        write_csv(sum_csv, summary, ["category", "count", "F1_avg", "BLEU1_avg"])
        write_csv(det_csv, details, ["q_idx", "category", "gold_answer", "prediction", "F1", "BLEU1", "source_folder"])
        
        # 打印整体统计
        for r in summary:
            print(f"Category {r['category']}: n={r['count']}, F1_avg={r['F1_avg']:.4f}, BLEU1_avg={r['BLEU1_avg']:.4f}")
        print(f"整体结果已保存到: {sum_csv}")
        print(f"详细结果已保存到: {det_csv}")
    
    print(f"\n完成! 已处理 {len(subdirs)} 个文件夹，合并 {len(all_data)} 条数据")

if __name__ == "__main__":
    main()