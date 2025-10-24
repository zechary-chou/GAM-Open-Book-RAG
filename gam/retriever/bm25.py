import os
import json
from typing import Dict, Any, List
from pyserini.search.lucene import LuceneSearcher

from gam.retriever.base import AbsRetriever
from gam.schemas import InMemoryPageStore, Hit


class BM25Retriever(AbsRetriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pages = None
        self.index = None

    def search(self, query_list: List[str], top_k: int = 10) -> List[List[Hit]]:
        results = []
        for query in query_list:
            hits = self.index_dict.search(query, k=top_k)
            tmp_results = {}
            for i in range(len(hits)):
                tmp_results[hits[i].docid] = {
                    'id': hits[i].docid,
                    'score': hits[i].score,
                    'content': Hit(
                        page_index=self.pages[int(hits[i].docid)].page_id,
                        snippet=self.pages[int(hits[i].docid)].content[:200],
                        source="keyword",
                        meta={"rank": i, "score": score[i]}
                    )
                }
            cur_scores = sorted(tmp_results.items(), key=lambda x: x[1]['score'], reverse=True)[:top_k]
            results.append([e[1]['content'] for e in cur_scores])
        return results

    def load(self):
        index_dir = self.config.get("index_dir")
        if os.path.exists(os.path.join(index_dir, "index")):
            try:
                self.pages = InMemoryPageStore.load(os.path.join(index_dir, "pages")).list_all()
                self.index = LuceneSearcher(os.path.join(index_dir, "index"))
            except Exception as e:
                print('cannot load index, error: ', e)

    def build(self, page_store: InMemoryPageStore):
        self.pages = page_store.list_all()
        index_dir = self.config.get("index_dir")

        os.makedirs(os.path.join(index_dir, "documents"), exist_ok=True)
        os.makedirs(os.path.join(index_dir, "index"), exist_ok=True)
        os.makedirs(os.path.join(index_dir, "pages"), exist_ok=True)
        doc_list = []
        for idx, page in enumerate(self.pages):
            doc_list.append({
                "id": str(idx),
                "content": (page.header + ' ' + page.content).strip()
            })
        with open(os.path.join(index_dir, "documents", "documents.jsonl"), "w") as f:
            for d in doc_list:
                f.write(json.dumps(d) + '\n')

        command = f"""python -m pyserini.index.lucene \
            --collection JsonCollection \
            --input {os.path.join(index_dir, "index")} \
            --index {os.path.join(index_dir, "documents")} \
            --generator DefaultLuceneDocumentGenerator \
            --threads {self.config.threads} \
            --storePositions --storeDocvectors --storeRaw"""

        exit_code = os.system(command)
        page_store.save(os.path.join(index_dir, "pages"))

        if exit_code == 0:
            self.index = LuceneSearcher(os.path.join(index_dir, "index"))
            print("build BM25 index success")
        else:
            print(f"build BM25 index error :{exit_code}")

    def update(self, page_store: InMemoryPageStore):
        self.build(page_store)
