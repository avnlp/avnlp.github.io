## Omega RAG

Omega RAG provides a framework to combine several advanced RAG techniques into a high-performing RAG pipeline.
Query Rewriting, Hyde, Adaptive retrieval (no retrieval, single-step, iterative retrieval), Correction by retrieval evaluation and confidence scoring, Unified active retrieval, Reranking, Citation generation, User feedback, Hybrid structured router, Scattered knowledge structurizer and Structured knowledge utilizer.  

Under active development.

## LLM-Finetuning
**Github**: [avnlp/llm-finetuning](https://github.com/avnlp/llm-finetuning)

Supervised Fine-Tuning (SFT) is the process of refining a pre-trained LLM using labeled data to align it with specific tasks or desired behaviors.
In this approach, the model learns to map inputs to desired outputs by minimizing a supervised loss function, typically cross-entropy.

The training data often consists of human-labeled examples, such as prompts paired with preferred responses. It is particularly useful for aligning general-purpose models with domain-specific requirements or enhancing performance on well-defined tasks.

P-tuning is a technique for fine-tuning large pre-trained models by optimizing a set of learnable prompt embeddings instead of updating the entire model’s parameters. It involves prepending task-specific, trainable prompt tokens to the input, which guides the model to generate desired outputs.

P-tuning is effective for adapting models to new tasks without requiring extensive retraining. It allows for faster, resource-efficient fine-tuning with minimal adjustments to the pre-trained model.

Prefix Tuning is a method for adapting pre-trained models to specific tasks by optimizing a set of fixed-length trainable prefix tokens added to the input sequence, without modifying the model's original parameters.

These prefix tokens, which are learned during training, steer the model's behavior and help it generate task-specific outputs. Like P-tuning, prefix tuning focuses on parameter efficiency, as it only adjusts the learned prefixes rather than the entire model.

It is particularly effective for large language models, offering a scalable and resource-efficient alternative to full fine-tuning. Prefix tuning allows for task-specific adaptation while preserving the core capabilities of the pre-trained model.

Prompt Tuning is a method for adapting pre-trained models by optimizing a set of task-specific prompt tokens, without modifying the model’s core parameters. These prompt tokens are prepended or appended to the input sequence, serving as a fixed, learnable set of parameters that guide the model's output generation.

In Self-RAG, the model learns a special “retrieve” token that tells it whether to call the retriever at each generation step. It can adaptively retrieve zero, one, or multiple passages as needed, rather than always fetching a fixed number.

Reflection Tokens for Critique - In addition to ordinary text tokens, the model is trained to emit “reflection” tokens at each segment:  
- ISREL (Is Retrieval Useful?): predicts if retrieved passages are relevant.
- ISSUP (Is Support Adequate?): evaluates whether retrieved passages sufficiently support the next generation.
- ISUSE (Is Output Useful?): judges the overall quality and utility of the continuation.

These tokens train the model to self-assess its retrieval and generation.

## Rankers

**Github:** [avnlp/rankers](https://github.com/avnlp/rankers)  
**Paper:** [LLM Rankers](https://github.com/avnlp/rankers/blob/main/paper/rankers.pdf)  

In RAG pipelines, top-K documents are first retrieved from a vector store. This initial retrieval prioritizes speed and recall, bringing a broad set of candidate documents. These documents are then sent to a re-ranker which orders these documents based on their relevance to the query. The re-ranker utilizes a LLM and is more computationally expensive than the retrieval stage. The re-ranker can use Pointwise, Pairwise, Listwise or Setwise ranking techniques. In this paper we have analyzed the performance of these techniques on the FiQA, SciFact, NFCorpus, TREC-19, and TREC-20 datasets. We evaluated multiple LLM-based rankers including Mistral, Phi-3, LLama-3, RankLlama (7B), and RankZephyr, comparing their effectiveness using the NDCG metric.

The results indicate a clear advantage for Listwise re-ranking as implemented by RankZephyr, followed by Pointwise re-ranking with RankLlama (7B). For other LLMs like LLama-3, Setwise techniques tended to perform slightly better than Pairwise, though both significantly enhanced retrieval quality over the baseline.

## RRF
**Github:** [avnlp/rrf](https://github.com/avnlp/rrf)

**Paper:** [Performance Evaluation of Rankers and RRF Techniques for Retrieval Pipelines](https://github.com/avnlp/rrf/blob/main/paper/rankers_rrf.pdf)

In the intricate world of Long-form Question Answering (LFQA) and Retrieval Augmented Generation (RAG), making the most of the LLM’s context window is paramount. Any wasted space or repetitive content limits the depth and breadth of the answers we can extract and generate. It’s a delicate balancing act to lay out the content of the context window appropriately.

With the addition of three rankers, viz., Diversity Ranker, Lost In The Middle Ranker, Similarity Rankers and RRF techniques, we aim to address these challenges and improve the answers generated by the LFQA/RAG pipelines. We have done a comparative study of adding different combinations of rankers in a Retrieval pipeline and evaluated the results on four metrics, viz., Normalized Discounted Cumulative Gain (NDCG), Mean Average Precision (MAP), Recall and Precision.



## LLM Blender

**Github:** [avnlp/llm-blender](https://github.com/avnlp/llm-blender) 

**Paper:** [LLM Ensembling: Haystack Pipelines with LLM-Blender](https://github.com/avnlp/llm-blender/blob/main/paper/llm_blender.pdf)

LLM-Blender is an ensembling framework designed to achieve consistently superior performance by combining the outputs of multiple language models (LLMs). This work focuses on integrating LLM-Blender with RAG pipelines to significantly improve the quality of generated text. 

A custom Haystack component, LLMBlenderRanker, has been implemented to integrate LLM-Blender with Haystack pipelines. The component utilizes the PairRanker module from the LLM-Blender framework, which compares each candidate with the input in a pairwise manner.

Different LLMs can generate subtly different texts, since they are trained on different datasets and tasks. By comparing each text in a pairwise manner, the component ranks and ensembles the text so it is robust to these subtle differences.

Ranking techniques like MLM-scoring, SimCLS, and SummaReranker focus on individually scoring each candidate based on the input text, but do not
compare candidates in a pairwise manner, which can lead to missing subtle differences between LLM outputs. 

Pipelines ensembling various LLMs, such as Mistral-7B, Llama-3-8B and Phi-3-mini, using the LLM-Blender were evaluated. The MixInstruct benchmark dataset was curated by the LLM-Blender authors to benchmark ensemble models for LLMs on instruction-following tasks. The pipelines were evaluated using the BERTScore, BARTScore, and BLEURT metrics on the MixInstruct and BillSum Datasets. 

The LLM-Blender framework, also introduced a GenFuser module, which fuses multiple LLM outputs to give a single condensed result. We found that the usage of the PairRanker alone gives better performance in RAG pipelines, as compared to using the PairRanker and GenFuser together.

We obtained better results on the MixInstruct and BillSum datasets than the results presented in the LLM Blender paper using more recent LLMs such as Mistral-7B, Llama-3-8B and Phi-3-mini.

## Vector-DB
**Github:** [avnlp/vectordb](https://github.com/avnlp/test-vectordb)

Designed and implemented pipelines demonstrating the use of various vector databases for Semantic Search, Metadata Filtering, Hybrid Search, Reranking, and Retrieval-Augmented Generation (RAG). The pipelines were made using the Pinecone, Weaviate, Chroma, Milvus and Qdrant vector databases.

We compare and contrast the functionality of the vector database with different pipelines for each technique.

Vector database pipelines were created using Langchain and Haystack to highlight the Hybrid Search, Metadata Filtering and Reranker.

Pipelines were developed for datasets such as TriviaQA, ARC, PopQA, FactScore, Earnings Calls, and SEC Filings.

Hybrid Search Pipelines: Created pipelines combining sparse and dense indexes by upserting data into each index separately. Hybrid Search enables a unified query approach that merges semantic (dense) and keyword (sparse) search for improved relevance in results.

Metadata Filtering Pipelines: Developed pipelines leveraging metadata fields associated with vectors to filter search results during query time. Metadata enhances vectors with contextual information, enabling more meaningful and precise filtering.

Reranker Pipelines: Implemented pipelines to rerank semantic search results, ensuring the most relevant results are prioritized and returned.



| Feature          | Milvus                                  | Weaviate                                  | Qdrant                                  | Pinecone                                  | Chromadb                                  |
|------------------|-----------------------------------------|-------------------------------------------|-----------------------------------------|-------------------------------------------|-------------------------------------------|
| **Indexes**      | Supports both sparse and dense vectors, using IVF for dense indexing and BM25 for sparse retrieval | Supports both sparse and dense vectors, using HNSW for dense and BM25 for sparse indexing | Supports both sparse and dense vectors, using HNSW for dense and hybrid search mechanisms for sparse | Supports only dense vectors, optimized for approximate nearest neighbor (ANN) search. Sparse vectors not supported | Supports only dense vectors with flat embeddings, optimized for in-memory search |
| **Hybrid Search** | BM25 + vector search using hybrid query modes | BM25 + vector search with `alpha` parameter for balance | BM25 + ANN search with structured filtering | Single sparse-dense index. Requires both sparse and dense query vectors | Not supported |
| **Partition**    | Uses partitions to separate data. Queries limited to a partition | Uses tenants for isolation. Queries limited to a tenant | Uses named collections for data separation. Queries filtered within collections | Uses namespaces to partition records. Queries limited to one namespace | Uses collections as namespaces. Queries directed to a collection |
| **Semantic Search** | Uses IVF, HNSW, and ANNOY for efficient vector retrieval | Vector-based retrieval. Results based on embedding similarity | Real-time vector similarity search with contextual relevance | Finds similar content using vector proximity. Supports metadata filtering | Stores and retrieves vector embeddings for similarity search |
| **Metadata Filtering** | SQL-like filtering with structured metadata fields | GraphQL-based filtering with hierarchical queries | Payload-based filtering with structured metadata | Dictionary-based metadata filtering attached to vectors | Key-value filtering using Pythonic expressions |

## Hyperparameter-Tuning
**Github:** [avnlp/hyperparameter-tuning](https://github.com/avnlp/hyperparameter-tuning)

**Paper:** [Optimizer Inclusions](https://github.com/avnlp/hyperparameter-tuning/blob/main/paper/optimizer_inclusions.pdf)

The choice of optimization algorithm for training Large Language Models (LLMs) significantly impacts both training speed and final predictive performance. In this paper, we demonstrate the critical importance of hyperparameter tuning protocols in optimizer comparisons for LLMs. 

Our research reveals that inclusion relationships between optimizers play a crucial role in practice and consistently predict optimizer performance. Contrary to conventional wisdom, we find that when carefully tuned, adaptive gradient methods such as Adam never underperform simpler optimizers like momentum or stochastic gradient descent (SGD). Our experiments show that more general optimizers consistently outperform their special cases, highlighting the practical significance of
inclusion relationships between optimizers. 

We illustrate the sensitivity of optimizer comparisons to hyperparameter tuning protocols by examining previous experimental evaluations. Our results demonstrate how changes in tuning protocols can significantly alter optimizer rankings for a given workload (model and dataset pair). Notably, as more effort is invested in tuning, optimizer rankings stabilize according to their inclusion relationships. To validate our findings, we conducted extensive hyperparameter tuning across three NLP tasks: Sentiment Analysis, Question Answering, and Text Summarization. We utilized well-known foundational language models and fine-tuned them on various
datasets for each task.

For Sentiment Analysis, we used Financial Phrasebank, StockTwits, and FinGPT-Sentiment datasets. Question Answering experiments were conducted on SQuAD, CoQA, and FIQA datasets, while Summarization tasks employed Multi-News and BillSum datasets. Using these fine-tuned models, we demonstrate the inclusion relationships for a range of optimizers, including Adam, RMSProp, Nesterov Accelerated Gradient (NAG), SGD with momentum, and vanilla SGD.
Our findings underscore that the differences between optimizers are entirely captured by their update rules and hyperparameters, with more expressive optimizers consistently outperforming their less expressive counterparts as hyperparameter tuning approaches optimality.



