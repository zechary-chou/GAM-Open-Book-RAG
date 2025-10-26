import os
import json
import numpy as np
from typing import Dict, Any, List
from FlagEmbedding import FlagAutoModel
from FlagEmbedding.abc.evaluation.utils import index as faiss_index
from FlagEmbedding.abc.evaluation.utils import search

from gam.retriever.base import AbsRetriever
from gam.schemas import InMemoryPageStore, Hit, Page


class IndexRetriever(AbsRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pages: List[Page] = []

    def load(self):
        index_dir = self.config.get("index_dir")
        try:
            self.page_store = InMemoryPageStore.load(os.path.join(index_dir, "pages"))
        except Exception as e:
            print('cannot load index, error: ', e)

    def build(self, page_store: InMemoryPageStore):
        self.page_store = page_store
        self.page_store.save(os.path.join(self.config.get("index_dir"), "pages"))

    def update(self, page_store: InMemoryPageStore):
        self.build(page_store)

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        hits: List[Hit] = []
        for query in query_list:
            # 尝试将查询解析为页面索引
            try:
                page_index = [int(idx.strip()) for idx in query.split(',') if idx.strip().isdigit()]
            except ValueError:
                # 如果解析失败，跳过这个查询
                continue
                
            for pid in page_index:
                p = self.page_store.get(pid)
                if not p:
                    continue
                hits.append(Hit(
                    page_id=str(pid),  # 使用页面索引作为page_id
                    snippet=p.content[:200],
                    source="page_index",
                    meta={}
                ))
        return [hits]  # 包装成 List[List[Hit]] 格式




