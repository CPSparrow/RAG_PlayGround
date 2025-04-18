from openai import OpenAI
import numpy as np
import faiss
import numpy as np
from typing import List


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


if __name__ == "__main__":
    embedding = EmbeddingModel()
    
    documents = [
        "Faiss is a library for efficient similarity search",
        "Facebook AI Research developed Faiss",
        "Approximate nearest neighbor search is useful for large datasets",
        "Vector databases power modern search applications"
    ]
    
    doc_embeddings = embedding(documents)
    
    # 构建索引
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    
    query = ["What is Faiss?", "what is a vector database?"]
    query_embedding = embedding(query)
    
    D, I = index.search(query_embedding, k=2)
    
    # 输出结果
    print("最相关的文档:")
    for query_index, item in enumerate(zip(I, D)):
        doc_list, score_list = item
        print(f"query {query_index + 1}:{query[query_index]}", "=====", sep='\n')
        for doc_index, score in zip(doc_list, score_list):
            print(f"{score:.3f}:{documents[doc_index]}")
