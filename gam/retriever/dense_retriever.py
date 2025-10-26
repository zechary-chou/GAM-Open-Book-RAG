import os
import json
import numpy as np
from typing import Dict, Any, List
from FlagEmbedding import FlagAutoModel
from FlagEmbedding.abc.evaluation.utils import index as faiss_index
from FlagEmbedding.abc.evaluation.utils import search

from gam.retriever.base import AbsRetriever
from gam.schemas import InMemoryPageStore, Hit, Page


class DenseRetriever(AbsRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pages = None
        self.index = None
        self.doc_emb = None
        self.model = FlagAutoModel.from_finetuned(
            config.get("model_name"),
            normalize_embeddings=config.get("normalize_embeddings", True),
            pooling_method=config.get("pooling_method", "cls"),
            trust_remote_code=config.get("trust_remote_code", True),
            query_instruction_for_retrieval=config.get("query_instruction_for_retrieval"),
            use_fp16=config.get("use_fp16", False),
            devices=config.get("devices", "cuda")
        )


    # ---------- 内部小工具 ----------
    def _index_dir(self) -> str:
        return self.config["index_dir"]

    def _pages_dir(self) -> str:
        return os.path.join(self._index_dir(), "pages")

    def _emb_path(self) -> str:
        return os.path.join(self._index_dir(), "doc_emb.npy")

    def _encode_pages(self, pages: List[Page]) -> np.ndarray:
        # 和 build() / update() 保持一致的编码方式
        texts = [(p.header + " " + p.content).strip() for p in pages]
        return self.model.encode_corpus(
            texts,
            batch_size=self.config.get("batch_size", 32),
            max_length=self.config.get("max_length", 512),
        )

    # ---------- 对外接口 ----------
    def load(self) -> None:
        """
        从磁盘恢复：
        - pages 快照
        - doc_emb.npy
        - faiss 索引
        """
        # 如果load失败，不抛死，只打印，这样ResearchAgent可以再走build()
        try:
            # 读向量
            self.doc_emb = np.load(self._emb_path())
            # 重建 index
            self.index = faiss_index(self.doc_emb)
            # 读 pages
            self.pages = InMemoryPageStore.load(self._pages_dir()).list_all()
        except Exception as e:
            print("DenseRetriever.load() failed, will need build():", e)

    def build(self, page_store: InMemoryPageStore) -> None:
        """
        全量重建向量索引。
        """
        os.makedirs(self._pages_dir(), exist_ok=True)

        # 1. 把当前 page_store 取出来
        self.pages = page_store.list_all()

        # 2. 全量编码
        self.doc_emb = self._encode_pages(self.pages)

        # 3. 建 faiss 索引
        self.index = faiss_index(self.doc_emb)

        # 4. 持久化
        page_store.save(self._pages_dir())
        np.save(self._emb_path(), self.doc_emb)

    def update(self, page_store: InMemoryPageStore) -> None:
        """
        增量更新：如果只是新增了一些 Page，或者后半段变了，
        我们就只重新编码“变化起点”之后的部分，而不是全量重算。
        """
        # 如果我们还没有 build 过，就直接走 build
        if not self.pages or self.doc_emb is None or self.index is None:
            self.build(page_store)
            return

        new_pages = page_store.list_all()
        old_pages = self.pages

        # 1. 找到第一个差异位置 diff_idx
        max_shared = min(len(new_pages), len(old_pages))
        diff_idx = max_shared  # 假设一开始完全一致
        for i in range(max_shared):
            if Page.equal(new_pages[i], old_pages[i]):
                continue
            diff_idx = i
            break

        # 2. 判断有没有实际变化
        changed = (diff_idx < max_shared) or (len(new_pages) != len(old_pages))
        if not changed:
            # 完全没变，直接返回
            return

        # 3. 我们保留前 diff_idx 段的老向量，后半段重新编码
        keep_emb = self.doc_emb[:diff_idx]

        tail_pages = new_pages[diff_idx:]
        tail_emb = self._encode_pages(tail_pages)

        new_doc_emb = np.concatenate([keep_emb, tail_emb], axis=0)

        # 4. 重新建 faiss 索引
        self.index = faiss_index(new_doc_emb)

        # 5. 持久化 + 刷内存
        page_store.save(self._pages_dir())
        np.save(self._emb_path(), new_doc_emb)

        self.pages = new_pages
        self.doc_emb = new_doc_emb

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        """
        输入: 多个query
        输出: 对应多个query的检索结果 (Hit 列表)
        """
        if self.index is None:
            # 如果还没 index（比如没调用 build/load），尝试load一下
            self.load()
            # 如果load也没成功，那 index 还是 None，就直接空返回
            if self.index is None:
                return [[] for _ in query_list]

        # 把所有 query 一起编码
        queries_emb = self.model.encode_queries(
            query_list,
            batch_size=self.config.get("batch_size", 32),
            max_length=self.config.get("max_length", 512),
        )

        # 用 FlagEmbedding 自带的 search() 查
        scores_list, indices_list = search(self.index, queries_emb, top_k)

        all_results: List[List[Hit]] = []
        for scores, indices in zip(scores_list, indices_list):
            hits_for_this_query: List[Hit] = []

            for rank, (idx, sc) in enumerate(zip(indices, scores)):
                idx_int = int(idx)
                if idx_int < 0 or idx_int >= len(self.pages):
                    continue
                page = self.pages[idx_int]
                snippet = (page.header + " " + page.content)[:200]

                hits_for_this_query.append(
                    Hit(
                        page_id=idx_int,          # 用统一的 int 索引
                        snippet=snippet,
                        source="vector",          # 和 planner/tool 名字保持一致
                        meta={"rank": rank, "score": float(sc)},
                    )
                )
            all_results.append(hits_for_this_query)

        return all_results