import json
from transformers import AutoTokenizer
from pathlib import Path

# CHANGE THIS to the model tokenizer you actually use
TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME,
    trust_remote_code=True
)

DATASETS = [
    "eval_400",
    "eval_1600",
    "eval_6400",
]

DATA_DIR = Path("./data/hotpotqa")

def build_prompt(example):
    """
    Adjust this if your actual prompt format differs.
    This mirrors typical HotpotQA memory-eval formatting.
    """
    question = example["question"]

    # Common field names in MemAgent-style datasets
    docs = example.get("documents") or example.get("contexts") or []

    context = "\n\n".join(
        f"[DOC {i}] {doc['text'] if isinstance(doc, dict) else doc}"
        for i, doc in enumerate(docs)
    )

    prompt = f"""You are given a question and a large memory context.

Context:
{context}

Question:
{question}

Answer:"""

    return prompt


for ds in DATASETS:
    path = DATA_DIR / f"{ds}.json"
    with open(path) as f:
        data = json.load(f)

    print(f"\n===== {ds} =====")

    # Check first 3 examples
    for i in range(3):
        example = data[i]
        prompt = build_prompt(example)

        tokens = tokenizer(prompt, return_tensors=None)["input_ids"]
        print(f"Example {i}: {len(tokens):,} tokens")
