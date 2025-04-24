import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal

import datasets
import faiss
import numpy as np
from datasets import Dataset
from openai import BadRequestError, OpenAI
from tqdm import tqdm


class EmbeddingModel:
    def __init__(self):
        self.client = OpenAI(
            api_key="empty",
            base_url="http://127.0.0.1:1234/v1"
        )
        self.model_name = "text-embedding-nomic-embed-text-v1.5-embedding"
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def embed_single_batch(self, text: list[str]) -> np.ndarray:
        assert len(text) <= 10, "too much items in input."
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return np.array([data.embedding for data in response.data], dtype=np.float32)
    
    def forward(self, text_list: List[str]) -> np.ndarray:
        assert isinstance(text_list, list), "text must be a list"
        
        batch_size = 10
        n_threads = 20
        
        if len(text_list) <= 100:
            embedding_list = []
            for i in range(0, len(text_list), batch_size):
                batch = text_list[i:i + batch_size]
                
                embedding_list.append(self.embed_single_batch(batch))
            return np.vstack(embedding_list)
        else:
            batch_list = [
                text_list[i:i + batch_size]
                for i in range(0, len(text_list), batch_size)
            ]
            with ThreadPoolExecutor(max_workers=20) as executor:
                embedding_list = list(tqdm(
                    executor.map(
                        lambda batch: self.embed_single_batch(batch),
                        batch_list
                    ),
                    total=len(text_list) // batch_size,
                    desc="Generating Embeddings",
                ))
            return np.vstack(embedding_list)


class LanguageModel:
    def __init__(self, rag_type: Literal["dense", "sparse", "cot", "none"]):
        self.client = OpenAI(
            api_key=os.getenv("ZHIPUAI_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        self.model_name = "glm-4-flash"
        self.rag_type = rag_type
        if rag_type == "cot":
            self.prompt = """You are required to answer a question and reply in short phrases.Below are some examples:\n[question]:Who proposed the theory of evolution by natural selection?\n[answer]:darwin\n[question]:Each specific polypeptide has a unique linear sequence of which acids?\n[answer]:amino\n[question]:A frameshift mutation is a deletion or insertion of one or more of what that changes the reading frame of the base sequence?\n[answer]:nucleotides\nNow please answer this question:\n[question]:"""
        elif rag_type == "none":
            self.prompt = ""
        elif rag_type == "dense" or rag_type == "sparse":
            self.prompt = """You are required to answer a question assisted by some documents and reply in short phrases.Below are some examples:\n[question]:Who proposed the theory of evolution by natural selection?\n[answer]:darwin\n[question]:Each specific polypeptide has a unique linear sequence of which acids?\n[answer]:amino\n[question]:A frameshift mutation is a deletion or insertion of one or more of what that changes the reading frame of the base sequence?\n[answer]:nucleotides\nNow please answer this question with the help of the documents after the question:\n[question]:"""
        else:
            raise NotImplementedError(f"RAG type {rag_type} not implemented")
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, question: str, k):
        if self.rag_type == "cot" or self.rag_type == "none":
            command = self.prompt + question
        elif self.rag_type == "dense":
            command = self.prepare_rag_prompt(self.prompt, question, k)
        else:
            raise NotImplementedError(f"rag type:{self.rag_type} not implemented")
        
        try:
            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": "Follow the instructions and give proper answer."},
                    {"role": "user", "content": command}
                ],
                top_p=0.7,
                temperature=0.3
            )
        except BadRequestError:
            return ""
        return response.choices[0].message.content
    
    def add_docs(self, docs: list[str]):
        if self.rag_type == "dense":
            self.docs = docs
            self.rag = DenseRAG(docs)
        elif self.rag_type == "cot" or self.rag_type == "none":
            raise ValueError(f"RAG type {self.rag_type} does not require docs input.")
        else:
            raise NotImplementedError(f"RAG type {self.rag_type} not implemented.")
    
    def prepare_rag_prompt(self, prompt, query: str, k: int):
        Distance, Index = self.rag.search([query], k=k)
        doc_str = "\nDocuments that may be helpful:\n"
        for item_index, index_doc_pair in enumerate(zip(Index[0], Distance[0])):
            doc_index, doc_score = index_doc_pair
            if doc_score >= 0.65:
                doc_str += f"[document #{item_index + 1}]:{self.docs[doc_index]}\n"
        return prompt + query + doc_str


class DenseRAG:
    def __init__(self, documents: list[str]):
        self.embedding = EmbeddingModel()
        self.file_path = "./storage/index.db"
        
        embeddings = self.embedding(documents)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self.index = index
    
    def __call__(self, *args, **kwargs):
        return search(*args, **kwargs)
    
    def search(self, query_list: list[str], k: int):
        assert isinstance(query_list, list), "query must be a list."
        query_emb = self.embedding(query_list)
        return self.index.search(query_emb, k=k)


def eval(
        rag_type: Literal["dense", "sparse", "cot", "none"],
        n_questions,
        n_docs=5,
        output_prefix="eval"
):
    lm = LanguageModel(rag_type)
    em = EmbeddingModel()
    data = get_data()
    
    if rag_type == "none":
        output_file = f"{output_prefix}_empty.json"
    elif rag_type == "cot":
        output_file = f"{output_prefix}_w_o_rag.json"
    elif rag_type == "dense":
        assert n_docs is not None, "n_docs must be provided."
        output_file = f"{output_prefix}_w_d_rag.json"
        lm.add_docs(data["docs"])
    else:
        raise NotImplementedError(f"RAG type {rag_type} not implemented")
    
    acc = 0
    embed_score_list = list()
    
    result = {
        "lm answer"   : list(),
        "ground truth": list(),
        "acc"         : list(),
        "embed_score" : list()
    }
    
    question_list = [data["question"][i] for i in range(n_questions)]
    ground_truth_list = [data["answer"][i] for i in range(n_questions)]
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        lm_answer_list = list(tqdm(
            executor.map(lambda query: lm(query, n_docs), question_list),
            total=n_questions,
            desc="Evaluating",
        ))
    
    for lm_ans, ground_truth in zip(lm_answer_list, ground_truth_list):
        if lm_ans.startswith("[answer]:"):
            lm_ans = lm_ans[len("[answer]:"):]
        result["lm answer"].append(lm_ans)
        result["ground truth"].append(ground_truth)
        
        if ground_truth in lm_ans:
            acc += 1
        emb = em([lm_ans, ground_truth])
        embed_score_list.append(emb[0] @ emb[1])
    
    acc /= n_questions
    embed_score = np.array(embed_score_list).mean()
    result["acc"].append(f"{acc:.3f}")
    result["embed_score"].append(f"{embed_score:.3f}")
    print(f"   Accuracy     : {acc:.3f}")
    print(f"Embedding score : {embed_score:.3f}")
    
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return acc, embed_score


def get_data():
    """
    return a dataset of 1000 lines. As for its content:
    Dataset({
        features: ['question', 'answer', 'docs'],
        num_rows: 1000
    })
    """
    dataset = datasets.load_dataset(
        path="./Datasets/sciq",
        cache_dir="./cache",
    )['test']
    dataset = dataset.remove_columns(['distractor3', 'distractor1', 'distractor2'])
    dataset = dataset.rename_columns({"correct_answer": "answer", "support": "docs"})
    return dataset


def rag_display():
    """
    simple demo, shows how faiss and embedding model works.
    """
    documents = [
        "Faiss is a library for efficient similarity search",
        "Facebook AI Research developed Faiss",
        "Approximate nearest neighbor search is useful for large datasets",
        "Vector databases power modern search applications"
    ]
    rag = DenseRAG(documents)
    
    k = 2
    query = ["What is Faiss?", "I like using databases."]
    
    Distance, Index = rag.search(query, k=k)
    
    print(f"最相关的前{k}个文档:")
    for query_index, item in enumerate(zip(Index, Distance)):
        doc_index_list, score_list = item
        print(f"query {query_index + 1}:{query[query_index]}", "=====", sep='\n')
        for doc_index, score in zip(doc_index_list, score_list):
            print(f"{score:.3f}:{documents[doc_index]}")


if __name__ == "__main__":
    n_questions = 1000
    n_docs = 5
    eval("cot", n_questions, n_docs)
    eval("dense", n_questions, n_docs)
    
    # rag_display()
