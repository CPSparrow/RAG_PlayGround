import json
import os
from typing import List

import datasets
import faiss
import numpy as np
from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm


class EmbeddingModel:
    def __init__(self):
        self.client = OpenAI(
            api_key="empty",
            base_url="http://127.0.0.1:1234/v1"
        )
        self.model_name = "text-embedding-nomic-embed-text-v1.5"
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, text: List[str]) -> np.ndarray:
        assert isinstance(text, list), "text must be a list"
        
        batch_size = 10
        embeddings = []
        
        for i in range(0, len(text), batch_size):
            batch = text[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            embeddings.append(
                np.array([data.embedding for data in response.data], dtype=np.float32)
            )
        
        return np.vstack(embeddings)


class LanguageModel:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        self.model_name = "glm-4-flash"
        self.zero_rag_prompt = """You are required to answer an question with simple answers.Below are some examples:\n[question]:Who proposed the theory of evolution by natural selection?\n[answer]:darwin\n[question]:Each specific polypeptide has a unique linear sequence of which acids?\n[answer]:amino\n[question]:A frameshift mutation is a deletion or insertion of one or more of what that changes the reading frame of the base sequence?\n[answer]:nucleotides\nNow please answer this question:\n[question]:"""
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, question):
        question = self.zero_rag_prompt + question
        response = self.client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "Follow the instructions and give proper answer."},
                {"role": "user", "content": question}
            ],
            top_p=0.7,
            temperature=0.3
        )
        return response.choices[0].message.content


class RAG:
    def __init__(self, documents):
        self.embedding = EmbeddingModel()
        self.file_path = "./storage/index.db"
        
        embeddings = self.embedding(documents)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.index = index
    
    def __call__(self, *args, **kwargs):
        return search(*args, **kwargs)
    
    def search(self, query, k=2):
        if isinstance(query, str):
            query = [query]
        query_emb = self.embedding(query)
        
        return self.index.search(query_emb, k=k)


def eval(n_questions=5, use_rag=False, output_prefix="eval"):
    data = load_dataset()
    lm = LanguageModel()
    em = EmbeddingModel()
    if use_rag:
        output_file = f"{output_prefix}_w_rag.json"
    else:
        output_file = f"{output_prefix}_w_o_rag.json"
    acc = 0
    embed_score_list = list()
    result = {
        "lm answer"   : list(),
        "ground truth": list(),
        "acc"         : list(),
        "embed_score" : list()
    }
    
    for i in tqdm(range(n_questions), ncols=80, desc="Evaluating"):
        question, ground_truth = data["question"][i], data["answer"][i]
        lm_ans = lm(question)
        result["lm answer"].append(lm_ans)
        result["ground truth"].append(ground_truth)
        
        if ground_truth in lm_ans:
            acc += 1
        embed_score_list.append(em([lm_ans])[0] @ em([ground_truth])[0])
    
    acc /= n_questions
    embed_score = np.array(embed_score_list).mean()
    result["acc"].append(f"{acc:.3f}")
    result["embed_score"].append(f"{embed_score:.3f}")
    print(f"   Accuracy     : {acc:.3f}")
    print(f"Embedding score : {embed_score:.3f}")
    
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return acc, embed_score


def load_dataset():
    """
    return a dataset of 1000 lines. As for its content:
    Dataset({
        features: ['question', 'answer', 'docs'],
        num_rows: 1000
    })
    """
    dataset = datasets.load_dataset(
        path="./sciq",
    )['test']
    dataset = dataset.remove_columns(['distractor3', 'distractor1', 'distractor2'])
    dataset = dataset.rename_columns({"correct_answer": "answer", "support": "docs"})
    return dataset


def rag_display():
    """
    simple demo, may be out of date.
    """
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
    eval(n_questions=20)
