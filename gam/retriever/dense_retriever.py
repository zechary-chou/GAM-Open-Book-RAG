import os
import json
import numpy as np
from typing import Dict, Any, List
from FlagEmbedding import FlagAutoModel
from FlagEmbedding.abc.evaluation.utils import index as faiss_index
from FlagEmbedding.abc.evaluation.utils import search

from gam.retriever.base import AbsRetriever
from gam.schemas import InMemoryPageStore, Hit


class DenseRetriever(AbsRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pages = None
        self.index = None
        self.doc_emb = None
        self.model = FlagAutoModel.from_finetuned(
            config.model_name, 
            query_instruction_for_retrieval=config.query_instruction_for_retrieval,
            use_fp16=config.use_fp16,
            devices=config.devices
        )

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        queries_emb = self.model.encode_queries(
            query_list, 
            batch_size=self.config.batch_size,
            max_length=self.config.max_length
        )
        scores, indices = search(self.index, queries_emb, top_k)
        results = []
        for score, indice in zip(scores, indices):
            results.append(
                [Hit(
                    page_id=self.pages[i].page_id,
                    snippet=self.pages[i].content[:200],
                    source="vector",
                    meta={"rank": i, "score": score[i]}
                )
                for i in range(len(score))]
            )
        return results

    def load(self):
        index_dir = self.config.index_dir
        try:
            self.doc_emb = np.load(os.path.join(index_dir, 'doc_emb.npy'))
            self.index = faiss_index(self.doc_emb)
            self.pages = InMemoryPageStore.load(os.path.join(index_dir, "pages")).list_all()
        except Exception as e:
            print('cannot load index, error: ', e)

    def build(self, page_store: InMemoryPageStore):
        os.makedirs(os.path.join(index_dir, "pages"), exist_ok=True)

        self.pages = page_store.list_all()
        self.doc_emb = self.model.encode_corpus([(page.header + ' ' + page.content).strip() for page in self.pages])
        self.index = faiss_index(self.doc_emb)

        index_dir = self.config.index_dir
        page_store.save(os.path.join(index_dir, "pages"))
        np.save(os.path.join(index_dir, 'doc_emb.npy'), self.doc_emb)
    
    def update(self, page_store: InMemoryPageStore):
        previous_pages = self.pages
        new_pages = page_store.list_all()
        max_idx = min(len(previous_pages), len(new_pages))
        for idx in range(max_idx):
            if previous_pages[idx] != new_pages[idx]:
                break
        
        if idx != max_idx or len(previous_pages) != len(new_pages):
            doc_emb = self.doc_emb[:idx]

            if idx == len(new_pages):
                self.doc_emb = doc_emb
            else:
                new_doc_emb = self.model.encode_corpus([e.content for e in new_pages[idx:]])
                self.doc_emb = np.concatenate([doc_emb, new_doc_emb], axis=0)

            self.index = faiss_index(self.doc_emb)
            self.pages = page_store.list_all()
            page_store.save(os.path.join(self.config.index_dir, "pages"))
            np.save(os.path.join(self.config.index_dir, 'doc_emb.npy'), self.doc_emb)
        else:
            print('no new pages to update')

