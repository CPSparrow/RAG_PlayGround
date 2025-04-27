# RAG PlayGround

> 这里预计存放学习RAG过程中编写的代码和笔记

## 一些比较好的文章

- **bm25** 的高效 python
  实现：[介绍链接](https://developer.volcengine.com/articles/7390577255279247371) / [GitHub链接](https://github.com/xhluca/bm25s)
- 介绍 **Rerank**: [知乎链接](https://zhuanlan.zhihu.com/p/676996307)
- 关于 **chunking**
  技术的一些介绍：[中文翻译](https://zhuanlan.zhihu.com/p/676979306) / [原文](https://www.pinecone.io/learn/chunking-strategies/)
- **RRF** :基于文档排名的一种技术：[网页链接](https://www.luxiangdong.com/2024/11/08/rrf/)
- 产品经理视角下的 RAG ：[知乎链接](https://zhuanlan.zhihu.com/p/8352563254)
- 如何评估 RAG / Rerank
  模型：来自LlamaIndex
  的[网页](https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)
- [BGE项目](https://bge-model.com/index.html):世界领先的 embedding / rerank
  模型：[GitHub链接](https://github.com/FlagOpen/FlagEmbedding/tree/master)
    - 该项目下的学习资料：[Tutorial](https://github.com/FlagOpen/FlagEmbedding/tree/master/Tutorials)

## 关于```faiss```库

- 索引方式的简单介绍：[知乎链接](https://zhuanlan.zhihu.com/p/530958094)
- 不同索引方法的比较：[CSDN文章](https://blog.csdn.net/uncle_ll/article/details/139819001)

## How to Use

1. 根据你使用的服务提供商的信息编辑 embedding 和 llm model对应的 ```api_key```, ```base_url``` 和 ```model_name```
   。也可以和本代码一样使用本地部署的模型。
2. 在终端运行 ```SimpleRAG.py``` ,评估结果会自动覆盖同目录下的 json 文件。

## 未经验证的技术

- ```Dspy```:DSPy(Demonstrate-Search-Predict with Python)是一个用于优化语言模型(LM)
  提示和权重的框架。**被认为**只需要定义运行步骤（比如“检索、总结、评估“），DSPy会自动学习如何最佳地组合这些步骤。

## 引用

这里使用的测试数据来自[sciq](https://huggingface.co/datasets/allenai/sciq)

```
@inproceedings{SciQ,
    title={Crowdsourcing Multiple Choice Science Questions},
    author={Johannes Welbl, Nelson F. Liu, Matt Gardner},
    year={2017},
    journal={arXiv:1707.06209v1}
}
```
