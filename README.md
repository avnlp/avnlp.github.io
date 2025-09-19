## Biothink

**Github**: [https://github.com/avnlp/biothink](https://github.com/avnlp/biothink)

- Developed BioThink, a framework featuring self-reflective reasoning, where the model explicitly structures its reasoning process within <think> </think> XML tags and performs self-evaluation using specialized tokens (Relevance, Grounding, Utility) to critically assess its own output quality and alignment on Bio-Medical question-answering.
- Trained the BioThink model on Qwen3-1.7B using QLoRA for efficient parameter adaptation and the GRPO algorithm for alignment, on a corpus of question-answers from MedInstruct, Mol-Instructions, PubMed abstracts, PubMed Central full texts, MedQA, and Clinical Guidelines datasets.
- Integrated five reward functions with GRPO to enforce generation of self-evaluation tokens (Relevance, Grounding, Utility) and ensure strict adherence to the required XML reasoning structure  tag presence, correct order, and syntactical validity.
- Implemented a robust answer correctness reward function and metric using DeepEval's GEval metric, configured with a custom LLM-as-a-Judge instruction tailored for Bio-Medical Question Answering.
- Systematically assessed model performance across seven metrics: XML Structure integrity for presence/order of all reasoning, answer, and self-eval tags, token accuracy - correct generation of Utility, Relevance, Groundness tokens, Answer Correctness - using custom GEval metric, Faithfulness - adherence to source context , and Answer Relevancy - question alignment using DeepEval's LLM-as-a-Judge metric.
- Further work to incorporate additional retrieval mechanisms into BioThink based on Adaptive RAG, Corrective RAG, RQ-RAG are ongoing.

## RAG-Model-Training

**Github**: [https://github.com/avnlp/rag-model-training](https://github.com/avnlp/rag-model-training)​

- Trained a Query Complexity Classifier for Adaptive RAG [(https://arxiv.org/abs/2403.14403)](https://arxiv.org/abs/2403.14403) on a combination of Musique, NQ, TriviaQA and HotpotQA datasets. The T5 model is trained to classify the query into Simple/Moderate/Complex based on whether the query can be answered without retrieval, single retrieval or multiple retrievals.
- Trained a Retrieval Evaluator for Corrective RAG [(https://arxiv.org/abs/2401.15884)](https://arxiv.org/abs/2401.15884) on the dataset provided by the authors. The T5 model is trained to classify documents as 'Correct', 'Ambiguous', or 'Incorrect' based on the input question and retrieved documents.
- Fine-tuned a 3-Stage Query Refinement model for RQ-RAG [(https://arxiv.org/abs/2404.00610)](https://arxiv.org/abs/2404.00610) (rewriting/decomposition/disambiguation). The Llama-3.2 model is trained to generate '[Rewritten_Query]', '[Decomposed_Query]' and '[Disambiguated_Query]' tokens.
- Trained a model for Self-Reflective RAG [(https://arxiv.org/abs/2310.11511)](https://arxiv.org/abs/2310.11511) on the Earnings-Call data. Created training data with the reflection tokens (Retrieval/Relevance/Grounding/Utility). Trained the model in two phases: critic training to evaluate retrieval and a generator training for using the critic's feedback to generate responses.
- Fine-tuned Agentic RAG [(https://github.com/dCaples/AutoDidact)](https://github.com/dCaples/AutoDidact) on TriviaQA with GRPO using multi-turn generation for autonomous missing-info detection, query rewriting, and validated tool-call generation on Llama-3-8B, leveraging LLM-as-a-Judge rewards.
- Fine-tuned ReZero Agentic RAG [(https://arxiv.org/abs/2504.11001)](https://arxiv.org/abs/2504.11001) on TriviaQA with GRPO for search retrying and query refinement on Llama-3-8B, leveraging a composite LLM-as-a-Judge reward function to enforce response structure.

## GRPO

**Github:** [https://github.com/avnlp/grpo](https://github.com/avnlp/grpo)

- Compared four implementations of GRPO from scratch, each demonstrating different approaches to the core algorithm while sharing common principles.
- Refactored the implementations to highlight the differences in the implementation of the core algorithm, reward functions, training frameworks, and reference model handling.
- The implementations share the Group Sampling, Reward Calculation, Advantage Normalization and Policy Update steps.
- Each implementation has different reward functions tailored to the task. There are Format Reward functions for enforcing XML-style reasoning and Correctness Reward functions to validate accuracy. The implementations use different training frameworks (e.g., DeepSpeed, pure PyTorch). Their approaches to generation - vLLM, HuggingFace transformers and batching also vary.
- Some implementations use a fixed reference model (via a separate server or a frozen copy) while others update the reference model periodically.

## LLM-Finetuning

**Github**: [https://github.com/avnlp/llm-finetuning](https://github.com/avnlp/llm-finetuning)

- Fine-tuned models for RAG with Reasoning on HotpotQA, FreshQA, and Musique datasets using QLoRA and GRPO on Llama3.2-3B, implementing four correctness reward functions - DeepEval's GEval with custom LLM-as-a-Judge for RAG, Summarization, Answer Relevancy, and Evidently AI's CorrectnessLLMEval and four format reward functions to enforce '<reasoning>' tags and multiline response compliance.
- Fine-tuned models for Math Reasoning on GSM8K using QLoRA and GRPO on Phi-4, Mistral-7B, Llama3.2-3B, Llama3.1-8B, and Gemma3-1B to generate step-by-step solutions, applying one correctness reward function and four format reward functions for 'reasoning' tags and multiline structure.
- Fine-tuned three models for Preference Alignment on UltraFeedback dataset using QLoRA: Zephyr-7B using DPO, Qwen2.5-1.5B via KTO, and Llama-3-8B via DPO, ORPO, and PPO (using LLM-Blender PairRM as reward model).
- Fine-tuned model for Question-Answering Preference Alignment on the WebGPT comparisons dataset using QLoRA with DPO and PPO (using LLM-Blender PairRM as reward model) on Llama-3-8B.
- Fine-tuned models for Adapter-based Supervised Fine-tuning using QLoRA, LoRA, DoRA, P-Tuning, and Prefix-Tuning on the ARC, Earnings Call, FactScore, PopQA, and TriviaQA datasets. Compared the performance of different adapter based supervised fine-tuning techniques.

## LLM Rankers

**Github:** [https://github.com/avnlp/rankers](https://github.com/avnlp/rankers)  
**Paper:** [LLM Rankers](https://github.com/avnlp/rankers/blob/main/paper/rankers.pdf)  

- Implemented Pairwise, Setwise and Listwise ranking techniques. Released modular ranker components for the Haystack LLM framework. The implementation for all the rankers utilized Structured-Generation and Pydantic validation for robust Zero-Shot LLM ranking.
- The Pairwise and Setwise rankers utilize efficient sorting methods (Heapsort and Bubblesort) to speed up inference.
- The Listwise ranker integrates with the RankLLM framework and supports LLMs specifically trained for ranking (such as RankGPT, RankLlama, and RankZephyr).
- Evaluated the performance of the ranking techniques on the FIQA, SciFact, NFCorpus, TREC-19, and TREC-20 datasets using the Mistral, Phi-3, and Llama-3 models.
- All rankers performed closely across all datasets. RankLlama and RankZephyr (with the Listwise ranker) achieved slightly better results than the other rankers. Among the base models, the Llama-3 model with the Setwise and Pairwise ranker performed the best.

## Pairwise Ranking Prompting (PRP)

**Github:** [https://github.com/avnlp/prp](https://github.com/avnlp/prp)

- Implemented the Pairwise-Ranking-Prompting ranking technique from the paper with three sorting strategies: All-pair, Heapsort, Sliding Window.
- 'PRP-allpair' enumerates all pairs and performs a global aggregation to generate a score for each document. It is highly insensitive to input ordering. It essentially ranks documents with win ratio.
- 'PRP-heapsort' uses the pairwise preferences from the LLM as a comparator with HeapSort. It favors lower computation complexity than PRP-allpair while also being largely insensitive to input orders.
- 'PRP-sliding_k' uses a sliding window that starts at the bottom of the initial ranking, compares pairs of documents, and swaps document pairs with a stride of 1. It has favorable time complexity but has high dependency on input order.
- Evaluated the performance of the ranker on the FIQA, SciFact, NFCorpus, TREC-19, and TREC-20 datasets using the Mistral, Phi-3, and Llama-3 models.
- The 'PRP-allpair' with the Llama-3 model performed the best across all datasets. 'PRP-sliding_k' and 'PRP-heapsort' perform similarly across all datasets.

## RRF

**Github:** [https://github.com/avnlp/rrf](https://github.com/avnlp/rrf)  
**Paper:** [Performance Evaluation of Rankers and RRF Techniques for Retrieval Pipelines](https://github.com/avnlp/rrf/blob/main/paper/rankers_rrf.pdf)

- Evaluated performance of ranking models in conjunction with Reciprocal Rank Fusion for fusing results from Hybrid Retrieval pipelines.
- Implemented Diversity Ranker that maximizes the overall diversity of the documents using embedding based similarity search. It uses a Sentence-Transformer model for embeddings.
- Implemented Lost In The Middle Ranker that positions the most relevant documents at the beginning and at the end of the prompt while placing the least relevant documents in the middle to overcome the Lost-in-middle problem of LLMs.
- Evaluated the performance of the rankers on the FIQA dataset using the `Instructor-XL` and `all-mpnet-base-v2` as the embedding models.
- Evaluated different combinations of Dense Retrieval, Hybrid Retrieval in conjunction with usage of the Diversity, Lost In The Middle and Similarity Rankers.
- The best performance was achieved with the Instructor-XL embedding model with Similarity Ranker, then a Diversity Ranker and Lost In The Middle Ranker.
- We found that new instruction-tuned embedding models like Instructor-XL outperform many combinations of rankers due to their ability to utilize data specific instructions.

## LLM Blender

**Github:** [https://github.com/avnlp/llm-blender](https://github.com/avnlp/llm-blender)  
**Paper:** [LLM Ensembling: Haystack Pipelines with LLM-Blender](https://github.com/avnlp/llm-blender/blob/main/paper/llm_blender.pdf)

- LLM-Blender is an ensembling framework designed to achieve consistently superior performance by combining the outputs of multiple language models (LLMs). This work focuses on integrating LLM-Blender with RAG pipelines to significantly improve the quality of generated text.
- A custom Haystack component, LLMBlenderRanker, has been implemented to integrate LLM-Blender with Haystack pipelines. The component utilizes the PairRanker module from the LLM-Blender framework, which compares each candidate with the input in a pairwise manner.
- Different LLMs can generate subtly different texts, since they are trained on different datasets and tasks. By comparing each text in a pairwise manner, the component ranks and ensembles the text so it is robust to these subtle differences.
- Ranking techniques like MLM-scoring, SimCLS, and SummaReranker focus on individually scoring each candidate based on the input text, but do not compare candidates in a pairwise manner, which can lead to missing subtle differences between LLM outputs.
- Pipelines ensembling various LLMs, such as Mistral-7B, Llama-3-8B and Phi-3-mini, using the LLM-Blender were evaluated. The MixInstruct benchmark dataset was curated by the LLM-Blender authors to benchmark ensemble models for LLMs on instruction-following tasks. The pipelines were evaluated using the BERTScore, BARTScore, and BLEURT metrics on the MixInstruct and BillSum Datasets.
- The performance of the RAG pipelines with the LLM-Blender Ranker was evaluated on the BillSum and MixInstruct datasets using the Mistral, Phi-3, and Llama-3 models on the BERTScore, BARTScore and BLEURT metrics.
- The newer models like Llama-3-8B, Phi-3-mini, and Mistral-7B significantly outperformed all the models used by the LLM Blender authors on all the three metrics: BERTScore, BARTScore and BLEURT on the MixInstruct dataset.

## Omega RAG

Omega RAG provides a framework to combine several advanced RAG techniques into a high-performing RAG pipeline.
Query Rewriting, Hyde, Adaptive retrieval (no retrieval, single-step, iterative retrieval), Correction by retrieval evaluation and confidence scoring, Unified active retrieval, Reranking, Citation generation, User feedback, Hybrid structured router, Scattered knowledge structurizer and Structured knowledge utilizer.  

Under active development.

## Vector-DB

**Github:** [https://github.com/avnlp/vectordb](https://github.com/avnlp/vectordb)

- Designed and implemented pipelines demonstrating the use of various vector databases for Semantic Search, Metadata Filtering, Hybrid Search, Reranking, and Retrieval-Augmented Generation (RAG). The pipelines were made using the Pinecone, Weaviate, Chroma, Milvus and Qdrant vector databases.
- We compare and contrast the functionality of the vector database with different pipelines for each technique.
- Vector database pipelines were created using Langchain and Haystack to highlight the Hybrid Search, Metadata Filtering and Reranker.
- Pipelines were developed for datasets such as TriviaQA, ARC, PopQA, FactScore, Earnings Calls, and SEC Filings.
- Hybrid Search Pipelines: Created pipelines combining sparse and dense indexes by upserting data into each index separately. Hybrid Search enables a unified query approach that merges semantic (dense) and keyword (sparse) search for improved relevance in results.
- Metadata Filtering Pipelines: Developed pipelines leveraging metadata fields associated with vectors to filter search results during query time. Metadata enhances vectors with contextual information, enabling more meaningful and precise filtering.
- Reranker Pipelines: Implemented pipelines to rerank semantic search results, ensuring the most relevant results are prioritized and returned.

| Feature          | Milvus                                  | Weaviate                                  | Qdrant                                  | Pinecone                                  | Chromadb                                  |
|------------------|-----------------------------------------|-------------------------------------------|-----------------------------------------|-------------------------------------------|-------------------------------------------|
| **Indexes**      | Supports both sparse and dense vectors, using IVF for dense indexing and BM25 for sparse retrieval | Supports both sparse and dense vectors, using HNSW for dense and BM25 for sparse indexing | Supports both sparse and dense vectors, using HNSW for dense and hybrid search mechanisms for sparse | Supports only dense vectors, optimized for approximate nearest neighbor (ANN) search. Sparse vectors not supported | Supports only dense vectors with flat embeddings, optimized for in-memory search |
| **Hybrid Search** | BM25 + vector search using hybrid query modes | BM25 + vector search with `alpha` parameter for balance | BM25 + ANN search with structured filtering | Single sparse-dense index. Requires both sparse and dense query vectors | Not supported |
| **Partition**    | Uses partitions to separate data. Queries limited to a partition | Uses tenants for isolation. Queries limited to a tenant | Uses named collections for data separation. Queries filtered within collections | Uses namespaces to partition records. Queries limited to one namespace | Uses collections as namespaces. Queries directed to a collection |
| **Semantic Search** | Uses IVF, HNSW, and ANNOY for efficient vector retrieval | Vector-based retrieval. Results based on embedding similarity | Real-time vector similarity search with contextual relevance | Finds similar content using vector proximity. Supports metadata filtering | Stores and retrieves vector embeddings for similarity search |
| **Metadata Filtering** | SQL-like filtering with structured metadata fields | GraphQL-based filtering with hierarchical queries | Payload-based filtering with structured metadata | Dictionary-based metadata filtering attached to vectors | Key-value filtering using Pythonic expressions |

## Hyperparameter-Tuning

**Github:** [https://github.com/avnlp/hyperparameter-tuning](https://github.com/avnlp/hyperparameter-tuning)  
**Paper:** [Optimizer Inclusions](https://github.com/avnlp/hyperparameter-tuning/blob/main/paper/optimizer_inclusions.pdf)

- The choice of optimization algorithm for training Large Language Models (LLMs) significantly impacts both training speed and final predictive performance.
- To illustrate the sensitivity of optimizer comparisons to hyperparameter tuning protocols, we conducted extensive hyperparameter tuning across three NLP tasks: Sentiment Analysis, Question Answering, and Text Summarization.
- For Sentiment Analysis, we used Financial Phrasebank, StockTwits, and FinGPT-Sentiment datasets. Question Answering experiments were conducted on SQuAD, CoQA, and FIQA datasets, while Summarization tasks employed Multi-News and BillSum datasets.
- We fine-tuned DistilBERT, BERT, and FinBERT models for Sentiment Analysis on the StockTwits and Financial PhraseBank dataset, while DistilBERT, BERT, RoBERTa were fine-tuned for Question Answering on the CoQA and SQuAD dataset. For Text Summarization, BART, DistillBART, and T5 models were fine-tuned on the BillSum and Multi-News dataset.
- Using these fine-tuned models, we demonstrated the inclusion relationships for a range of optimizers, including Adam, RMSProp, Nesterov Accelerated Gradient (NAG), SGD with momentum, and vanilla SGD.
