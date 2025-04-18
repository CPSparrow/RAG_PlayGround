from openai import OpenAI
import numpy as np
import faiss
from typing import List
from datasets import Dataset
import datasets
import os


class EmbeddingModel:
    def __init__(self):
        self.client = OpenAI(
            api_key="empty",
            base_url="http://127.0.0.1:1234/v1"
        )
        self.model = "text-embedding-nomic-embed-text-v1.5"
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, text: List[str]) -> np.ndarray:
        assert isinstance(text, list), "text must be a list"
        
        batch_size = 10
        embeddings = []
        
        for i in range(0, len(text), batch_size):
            batch = text[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            embeddings.append(
                np.array([data.embedding for data in response.data], dtype=np.float32)
            )
        
        return np.vstack(embeddings)


class RAG:
    def __init__(self, documents):
        self.embedding = EmbeddingModel()
        self.file_path = "./storage/index.db"
        
        embeddings = self.embedding(documents)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.index = index
        # faiss.write_index(index, self.file_path)
    
    def __call__(self, *args, **kwargs):
        return search(*args, **kwargs)
    
    # def load(self):
    #     return faiss.read_index(self.file_path)
    #
    # def create(self, text: List[str]):
    #     pass
    
    def search(self, query, k=2):
        if isinstance(query, str):
            query = [query]
        query_emb = self.embedding(query)
        
        return self.index.search(query_emb, k=k)


def eval():
    pass


def load_dataset():
    """
    return a dataset of 1000 lines. As for its content:
    Dataset({
        features: ['question', 'answer', 'docs'],
        num_rows: 1000
    })
    """
    base_dir = "/home/coder/Documents/CodeAndFiles/Corpus"
    cache_dir = os.path.join(base_dir, "cache")
    
    dataset = datasets.load_dataset(
        path=os.path.join(base_dir, "sciq"),
        cache_dir=cache_dir,
    )['test']
    dataset = dataset.remove_columns(['distractor3', 'distractor1', 'distractor2'])
    dataset = dataset.rename_columns({"correct_answer": "answer", "support": "docs"})
    return dataset


def rag_display():
    documents = data["docs"]
    rag = RAG(data['docs'])
    
    query = ["What is a phosphatase?", "what is the study of ecosystems?"]
    
    Distance, Index = rag.search(query, k=5)
    
    print("最相关的文档:")
    for query_index, item in enumerate(zip(Index, Distance)):
        doc_index_list, score_list = item
        print(f"query {query_index + 1}:{query[query_index]}", "=====", sep='\n')
        for doc_index, score in zip(doc_index_list, score_list):
            print(f"{score:.3f}:{documents[doc_index]}")


if __name__ == "__main__":
    embedding = EmbeddingModel()
    data = load_dataset()
    # rag_display()
