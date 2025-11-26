# General Agentic Memory (GAM)

A general memory system for agents, powered by deep-research

[中文文档](README_CN.md) | English

<h5 align="center"> 🎉 If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

**General Agentic Memory (GAM)** provides a next-generation memory framework for AI agents, combining long-term retention with dynamic reasoning. Following the Just-in-Time (JIT) principle, it preserves full contextual fidelity offline while performing deep research online to build adaptive, high-utility context. With its dual-agent architecture—Memorizer and Researcher—GAM integrates structured memory with iterative retrieval and reflection, achieving state-of-the-art performance across LoCoMo, HotpotQA, LongBench v2, and LongCodeBench benchmarks.

- **Paper**: 
- **Website**: 
- **Documentation**: 
- **YouTube Video**: 

<span id='features'/>

## ✨ Key Features

* 🧠 **Just-in-Time (JIT) Memory Optimization**
</br> Unlike conventional Ahead-of-Time (AOT) systems, GAM performs intensive Memory Deep Research at runtime, dynamically retrieving and synthesizing high-utility context to meet real-time agent needs.

* 🔍 **Dual-Agent Architecture: Memorizer & Researcher**
</br> A cooperative framework where the Memorizer constructs structured memory from raw sessions, and the Researcher performs iterative retrieval, reflection, and summarization to deliver precise, adaptive context.

* 🚀 **Superior Performance Across Benchmarks**
</br> Achieves state-of-the-art results on LoCoMo, HotpotQA, LongBench v2, and LongCodeBench, surpassing prior systems such as A-MEM, Mem0, and MemoryOS in both F1 and BLEU-1 metrics.

* 🧩 **Modular & Extensible Design**
</br> Built to support flexible plug-ins for memory construction, retrieval strategies, and reasoning tools—facilitating easy integration into multi-agent frameworks or standalone LLM deployments.

* 🌐 **Cross-Model Compatibility**
</br> Compatible with leading LLMs such as GPT-4, GPT-4o-mini, and Qwen2.5, supporting both cloud-based and local deployments for research or production environments.

<span id='news'/>

## 📣 Latest News

- **2025-11**: Released GAM framework with modular evaluation suite
- **2025-11**: Support for HotpotQA, NarrativeQA, LoCoMo, and RULER benchmarks

## 📑 Table of Contents

* [✨ Features](#features)
* [🔥 News](#news)
* [🏗️ Project Structure](#structure)
* [🎯 Quick Start](#quick-start)
* [🔬 Reproducing Paper Results](#reproduce)
* [📖 Documentation](#doc)
* [🌟 Citation](#cite)
* [🤝 Community](#community)

<span id='structure'/>

## 🏗️ System Architecture

![logo](./assets/GAM-memory.png)

## 🏗️ Project Structure

```
general-agentic-memory/
├── gam/                          # Core GAM package
│   ├── __init__.py
│   ├── agents/                   # Agent implementations
│   │   ├── memory_agent.py      # MemoryAgent - memory construction
│   │   └── research_agent.py    # ResearchAgent - deep research
│   ├── generator/                # LLM generators
│   │   ├── openai_generator.py  # OpenAI API generator
│   │   └── vllm_generator.py    # VLLM local generator
│   ├── retriever/                # Retrievers
│   │   ├── index_retriever.py   # Index retrieval
│   │   ├── bm25.py              # BM25 keyword retrieval
│   │   └── dense_retriever.py   # Dense semantic retrieval
│   ├── prompts/                  # Prompt templates
│   ├── schemas/                  # Data models
│   └── config/                   # Configuration management
├── eval/                         # Evaluation suite
│   ├── __init__.py
│   ├── run.py                   # Unified CLI entry
│   ├── README.md                # Evaluation documentation
│   ├── QUICKSTART.md            # Quick start guide
│   ├── datasets/                # Dataset adapters
│   │   ├── base.py             # Base evaluation class
│   │   ├── hotpotqa.py         # HotpotQA multi-hop QA
│   │   ├── narrativeqa.py      # NarrativeQA narrative QA
│   │   ├── locomo.py           # LoCoMo conversation memory
│   │   └── ruler.py            # RULER long-context eval
│   └── utils/                   # Evaluation utilities
│       ├── chunking.py         # Text chunking
│       └── metrics.py          # Evaluation metrics
├── scripts/                      # Shell scripts
│   ├── eval_hotpotqa.sh
│   ├── eval_narrativeqa.sh
│   ├── eval_locomo.sh
│   ├── eval_ruler.sh
│   └── eval_all.sh
├── examples/                     # Usage examples
│   └── quickstart/              # Quick start examples
│       ├── README.md            # Examples documentation
│       ├── basic_usage.py       # Basic usage example
│       └── model_usage.py       # Model selection example
├── assets/                       # Resource files
├── docs/                         # Documentation
├── setup.py                      # Installation config
├── pyproject.toml               # Modern project config
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

<span id='quick-start'/>

## 🎯 Quick Start

### 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/VectorSpaceLab/general-agentic-memory.git
cd general-agentic-memory

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 💡 Basic Usage

```python
import os
from gam import (
    MemoryAgent,
    ResearchAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    InMemoryMemoryStore,
    InMemoryPageStore,
    DenseRetriever,
    DenseRetrieverConfig,
)

# 1. Configure and create generator
gen_config = OpenAIGeneratorConfig(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3
)
generator = OpenAIGenerator(gen_config)

# 2. Create memory and page stores
memory_store = InMemoryMemoryStore()
page_store = InMemoryPageStore()

# 3. Create MemoryAgent
memory_agent = MemoryAgent(
    generator=generator,
    memory_store=memory_store,
    page_store=page_store
)

# 4. Memorize documents
documents = [
    "Artificial Intelligence is a branch of computer science...",
    "Machine Learning is a subset of AI...",
    "Deep Learning uses neural networks..."
]

for doc in documents:
    memory_agent.memorize(doc)

# 5. Get memory state
memory_state = memory_agent.get_memory_state()
print(f"Built {len(memory_state.events)} memory events")

# 6. Create ResearchAgent for Q&A
retriever_config = DenseRetrieverConfig(
    model_path="BAAI/bge-base-en-v1.5"
)
retriever = DenseRetriever(
    config=retriever_config,
    memory_store=memory_store,
    page_store=page_store
)

research_agent = ResearchAgent(
    generator=generator,
    retriever=retriever
)

# 7. Perform research
result = research_agent.research(
    question="What is the difference between ML and DL?",
    top_k=3
)

print(f"Answer: {result.final_answer}")
```

### 📚 Complete Examples

For detailed examples and advanced usage:
- [`examples/quickstart/basic_usage.py`](./examples/quickstart/basic_usage.py) - Complete workflow with memory building and research
- [`examples/quickstart/model_usage.py`](./examples/quickstart/model_usage.py) - Model selection and configuration
- [`examples/quickstart/README.md`](./examples/quickstart/README.md) - Examples documentation

<span id='reproduce'/>

## 🔬 How to Reproduce the Results in the Paper

We provide a complete evaluation framework to reproduce the experimental results in the paper.

### Quick Start

```bash
# 1. Prepare datasets
mkdir -p data
# Place your datasets in the data/ directory

# 2. Set environment variables
export OPENAI_API_KEY="your_api_key_here"

# 3. Run evaluations
# HotpotQA
bash scripts/eval_hotpotqa.sh --data-path data/hotpotqa.json

# NarrativeQA
bash scripts/eval_narrativeqa.sh --data-path narrativeqa --max-samples 100

# LoCoMo
bash scripts/eval_locomo.sh --data-path data/locomo.json

# RULER
bash scripts/eval_ruler.sh --data-path data/ruler.jsonl --dataset-name niah_single_1

# Or run all evaluations
bash scripts/eval_all.sh
```

### Using Python CLI

```bash
python -m eval.run \
    --dataset hotpotqa \
    --data-path data/hotpotqa.json \
    --generator openai \
    --model gpt-4 \
    --retriever dense \
    --max-samples 100
```

### Documentation

For complete evaluation documentation:
- [eval/README.md](./eval/README.md) - Evaluation framework guide
- [eval/QUICKSTART.md](./eval/QUICKSTART.md) - Quick start guide

### Supported Datasets

| Dataset | Task Type | Metrics | Documentation |
|---------|-----------|---------|---------------|
| **HotpotQA** | Multi-hop QA | F1 | [View](./eval/datasets/hotpotqa.py) |
| **NarrativeQA** | Narrative QA | F1 | [View](./eval/datasets/narrativeqa.py) |
| **LoCoMo** | Conversation Memory | F1, BLEU-1 | [View](./eval/datasets/locomo.py) |
| **RULER** | Long Context | Accuracy | [View](./eval/datasets/ruler.py) |

<span id='doc'/>

## 📖 Documentation

More detailed documentation is coming soon 🚀. Check these resources in the meantime:

- [Examples Documentation](./examples/quickstart/README.md) - Usage examples and tutorials
- [Evaluation Guide](./eval/README.md) - Evaluation framework documentation
- [Quick Start Guide](./eval/QUICKSTART.md) - Quick start for evaluations

<span id='cite'/>

## 📣 Citation

**If you find this project useful, please consider citing our paper:**

```bibtex
```

<span id='community'/>

## 🤝 Community

### 🎯 Contact Us

- GitHub Issues: [Report bugs or request features](https://github.com/VectorSpaceLab/general-agentic-memory/issues)
- Email: your-email@example.com

### 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=VectorSpaceLab/general-agentic-memory&type=Date)](https://star-history.com/#VectorSpaceLab/general-agentic-memory&Date)

### 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the authors of the following datasets:
- HotpotQA
- NarrativeQA
- LoCoMo
- RULER

## Disclaimer

This is a research project. Please use it responsibly and ethically.

---

<p align="center">
Made with ❤️ by the GAM Team
</p>
