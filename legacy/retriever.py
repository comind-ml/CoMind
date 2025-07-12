import os
import json
import logging

import numpy as np
import requests
from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from utils.config import Config

logger = logging.getLogger("graphml")
embed_model = None
tokenizer = None

def get_embedding(inputs: List[str], batch_size=128) -> Tensor:
    if len(inputs) == 0:
        return torch.tensor([])
    
    request_url = "http://localhost:8000/get_embedding"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(inputs)
    response = requests.post(request_url, headers=headers, data=data, timeout=36000)

    if response.status_code != 200:
        raise ValueError(f"Failed to get embedding: {response.text}")

    embedding = torch.tensor(json.loads(response.json()))
    return embedding

class Retriever:
    def _load_data_dir(self, doc_dir: Path):
        if not doc_dir.exists() or not doc_dir.is_dir():
            raise ValueError(f"Document directory {doc_dir} does not exist or is not a directory.")

        doc_info_path = doc_dir / "info.json"
        if not doc_info_path.exists():
            raise ValueError(f"Metadata {doc_info_path} does not exist in the document directory.")
        
        with open(doc_info_path, "r") as f:
            self.info = json.load(f)
        
        config_keys = list(self.info.keys())
        doc_names = doc_dir.glob("*.txt")

        self.documents = []
        self.documents_by_id = {}
        for doc in doc_names:
            if doc.stem not in config_keys:
                raise ValueError(f"Document {doc.stem} is not listed in the info.json file. Perhaps the document folder is corrupted.")
            if "votes" not in self.info[doc.stem]:
                raise ValueError(f"Document {doc.stem} does not have a 'votes' key in the configuration file. ")

            with open(doc_dir / doc.name, "r") as f:
                text = f.read()
            if len(text) == 0:
                continue

            metadata = {
                "votes": self.info[doc.stem]["votes"],
                "title": self.info[doc.stem]["title"],
                "id": doc.stem,
            }

            if "is_lower_better" in self.info[doc.stem]:
                self.is_lower_better = self.info[doc.stem]["is_lower_better"]
                metadata["bestPublicScore"] = self.info[doc.stem]["bestPublicScore"]
            

            self.documents_by_id[doc.stem] = text
            self.documents.append(Document(
                page_content=text, 
                metadata=metadata
            ))

    def __init__(self, cfg: Config, doc_dir: Path):
        self.cfg = cfg
        self.doc_dir = doc_dir
        self.is_lower_better = None

        logger.info(f"Loading document directory {doc_dir}...")

        self._load_data_dir(doc_dir)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.retriever.max_chunk_size,
            chunk_overlap=cfg.retriever.chunk_overlap,
        )

        self.split_docs = []
        for doc in self.documents:
            for chunk in splitter.split_documents([doc]):
                chunk.metadata["votes"] = doc.metadata["votes"]
                self.split_docs.append(chunk)

        content = [split_doc.page_content for split_doc in self.split_docs]
        # self.embeddings = get_embedding(content)

        logger.info(f"Document directory {doc_dir} loaded with {len(self.documents)} documents.")
    
    def get_detailed_instruct(self, query: str) -> str:
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        return f'Instruct: {task}\nQuery: {query}'

    def get_best_docs(self, k = 10) -> List[str]:
        if self.is_lower_better is None:
            return []

        doc_with_scores = [doc for doc in self.documents if doc.metadata["bestPublicScore"] is not None]
        if len(doc_with_scores) == 0:
            return []

        sorted_docs = sorted(
            doc_with_scores, 
            key=lambda x: x.metadata["bestPublicScore"], 
            reverse=not self.is_lower_better
        )

        k = min(k, len(sorted_docs))
        results = []
        for doc in sorted_docs[:k]:
            results.append(doc.page_content)
        return results
    
    def get_hotest_docs(self, k = 10) -> List[str]:
        """Get the top k hottest documents based on votes."""
        sorted_docs = sorted(
            self.documents, 
            key=lambda x: x.metadata["votes"], 
            reverse=True
        )

        k = min(k, len(sorted_docs))
        
        results = []
        for doc in sorted_docs[:k]:
            results.append(doc.page_content)
        
        return results


    def _calc_score(self, raw_score, vote):
        return raw_score * np.clip(np.log(np.log(vote + 1) + 1), 0.9, 1.1)
    
    
    def get_relevant_docs(self, query: str, by: str="content") -> List[str]:
        if len(self.documents) == 0:
            return []
        
        if by not in ["content", "id"]:
            raise ValueError(f"Invalid value for 'by': {by}. Expected 'content' or 'id'.")

        if by == "content":
            query = self.get_detailed_instruct(query)
            q_embeddings = get_embedding([query])

            scores = (q_embeddings @ self.embeddings.T)[0]
            raw_results = [(doc, self._calc_score(scores[id_], doc.metadata["votes"])) for id_, doc in enumerate(self.split_docs)]

            scored_results = sorted(
                raw_results, 
                key=lambda x: x[1],
                reverse=True
            )[:self.cfg.retriever.k]

            results, doc_ids = [], []
            for doc, _ in scored_results:
                if doc.metadata["id"] not in doc_ids:
                    results.append(self.documents_by_id[doc.metadata["id"]])
                    doc_ids.append(doc.metadata["id"])
                if len(results) >= self.cfg.retriever.k:
                    break
            
        else:
            ids = query.split("\n")
            results = []
            for id in ids:
                if id in self.documents_by_id:
                    results.append(self.documents_by_id[id])

        return results







    
