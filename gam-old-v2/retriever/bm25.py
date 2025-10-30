import os, json, subprocess, shutil
from typing import Dict, Any, List

try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    LuceneSearcher = None  # type: ignore

from gam.retriever.base import AbsRetriever
from gam.schemas import InMemoryPageStore, Hit, Page


class BM25Retriever(AbsRetriever):
    """
        关键词检索器 (BM25 / Lucene)
        config 需要:
        {
            "index_dir": "xxx",   # 用来放 index/ 和 pages/
            "threads": 4
        }
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if LuceneSearcher is None:
            raise ImportError("BM25Retriever requires pyserini to be installed")
        self.index_dir = self.config["index_dir"]
        self.searcher: LuceneSearcher | None = None
        self.pages: List[Page] = []

    def _pages_dir(self):
        return os.path.join(self.index_dir, "pages")

    def _lucene_dir(self):
        return os.path.join(self.index_dir, "index")

    def _docs_dir(self):
        return os.path.join(self.index_dir, "documents")

    def load(self) -> None:
        # 尝试从磁盘恢复
        if not os.path.exists(self._lucene_dir()):
            raise RuntimeError("BM25 index not found, need build() first.")
        self.pages = InMemoryPageStore.load(self._pages_dir()).load()
        self.searcher = LuceneSearcher(self._lucene_dir())  # type: ignore

    def build(self, page_store: InMemoryPageStore) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self._docs_dir(), exist_ok=True)

        # 1. dump pages -> documents.jsonl (pyserini需要 id + contents)
        pages = page_store.load()
        docs_path = os.path.join(self._docs_dir(), "documents.jsonl")
        with open(docs_path, "w", encoding="utf-8") as f:
            for i, p in enumerate(pages):
                text = (p.header + " " + p.content).strip()
                json.dump({"id": str(i), "contents": text}, f, ensure_ascii=False)
                f.write("\n")

        # 2. 清理旧的 lucene index
        if os.path.exists(self._lucene_dir()):
            shutil.rmtree(self._lucene_dir())
        os.makedirs(self._lucene_dir(), exist_ok=True)

        # 3. 调 pyserini 构建 Lucene 索引
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", self._docs_dir(),
            "--index", self._lucene_dir(),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(self.config.get("threads", 1)),
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        subprocess.run(cmd, check=True)

        # 4. 把 pages 也固化到磁盘，供 load() / search() 反查
        # 创建临时 PageStore 实例来保存
        temp_page_store = InMemoryPageStore(dir_path=self._pages_dir())
        temp_page_store.save(pages)
        
        # 5. 更新内存镜像
        self.pages = pages
        self.searcher = LuceneSearcher(self._lucene_dir())  # type: ignore

    def update(self, page_store: InMemoryPageStore) -> None:
        # Lucene 没有好用的“增量追加+可删改文档”的轻量接口（有但复杂）；
        # 对现在这个原型我们可以直接全量重建，保持简单可靠。
        self.build(page_store)

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        if self.searcher is None:
            # 容错：如果忘了 load/build
            self.load()

        results_all: List[List[Hit]] = []
        for q in query_list:
            q = q.strip()
            if not q:
                results_all.append([])
                continue

            hits_for_q = []
            py_hits = self.searcher.search(q, k=top_k)
            for rank, h in enumerate(py_hits):
                # h.docid 是字符串 id
                idx = int(h.docid)
                if idx < 0 or idx >= len(self.pages):
                    continue
                page = self.pages[idx]
                snippet = page.content
                hits_for_q.append(
                    Hit(
                        page_id=str(idx),
                        snippet=snippet,
                        source="keyword",
                        meta={"rank": rank, "score": float(h.score)}
                    )
                )
            results_all.append(hits_for_q)
        return results_all
