# AVNLP

Retrieval &amp; Ranking · RAG · Agents · LLM Training &amp; RL Alignment · Domain-Specific AI

**GitHub:** [github.com/avnlp](https://github.com/avnlp)

---

## Agentic Graph RAG for Medical Diagnosis

**GitHub:** [https://github.com/avnlp/agentic-med-diag](https://github.com/avnlp/agentic-med-diag)  
**Documentation:** [https://deepwiki.com/avnlp/agentic-med-diag](https://deepwiki.com/avnlp/agentic-med-diag)  
**Status:** Code coming soon

- Designed an agentic, multi-hop Graph RAG system for medical question answering that reasons across a medical knowledge graph to resolve complex clinical questions.
- Architected the system as a hierarchical **LangGraph** state machine: a parent graph fans out to two parallel subgraphs - a semantic channel and a relational channel, each with conditional loop edges - then synthesizes their outputs into a final answer.
- Decomposed each question into semantic sub-queries (with `#N` back-references) and relational SPO triple queries (with `Entity#N` placeholders), driving two iterative retrieve-and-reason loops: the semantic channel chains sub-query grounding → GraphRAG retrieval → semantic filter → summary → sub-answer → logic drafting → evidence verification → conditional expansion, while the relational channel runs SPO triple queries → KG filter → triplet summaries → sub-answer.
- Backed the two channels with **Neo4j** for the entity/relationship knowledge graph and **Milvus** for dense semantic retrieval over document chunks.
- Integrated four runtime-switchable Graph RAG backends, each with a custom `context_filter` that splits retrieved context across the channels: **LightRAG** (dual-level local/global KG in hybrid mode), **MiniRAG** (lightweight, Neo4j-free "light" mode), **PathRAG** (two-tier hierarchy returning reasoning paths between entities), and **HyperGraphRAG** (hyperedges linking multiple entities to capture group clinical relationships).
- Defined an evaluation protocol benchmarking all four pipelines against their stock backends on **HealthBench**, **MedCaseReasoning**, **MetaMedQA**, and **PubMedQA** with DeepEval's Contextual Recall, Contextual Precision, Contextual Relevancy, Answer Relevancy, and Faithfulness.

## BioThink: Self-Reflective Reasoning for Biomedical Question Answering

**GitHub:** [https://github.com/avnlp/biothink](https://github.com/avnlp/biothink)  
**Documentation:** [https://deepwiki.com/avnlp/biothink](https://deepwiki.com/avnlp/biothink)  
**Model:** [BioThink-Qwen3-1.7B](https://huggingface.co/avnlp/BioThink-Qwen3-1.7B)  
**Dataset:** [self_biorag_processed](https://huggingface.co/datasets/avnlp/self_biorag_processed)  

- Developed **BioThink**, a self-reflective biomedical question-answering framework inspired by Self-RAG and built on Self-BioRAG, that fine-tunes **Qwen3-1.7B** to interleave reasoning, answering, and self-critique in a single structured generation.
- Trained the model to emit a fixed XML schema: step-by-step reasoning in `<think>`, a concise `<answer>`, and three self-reflection signals - `<contextual-relevance>` (`[Relevant]`/`[Irrelevant]`), `<answer-utility>` (`[Utility:1]`–`[Utility:5]`), and `<groundness>` (`[Fully supported]`/`[Partially supported]`/`[No support/Contradictory]`).
- Aligned the model with **GRPO** (using QLoRA and Unsloth for efficient fine-tuning) under six reward functions: a Correctness reward (DeepEval's GEval with a custom biomedical LLM-as-a-Judge), Utility/Relevance/Groundness token rewards, an XML Structure reward (tag presence and open/close), and a Structure Order reward enforcing correct tag order with no stray text.
- Processed the Self-BioRAG dataset into question, answer, and context fields plus groundness, relevance, and utility labels, releasing it as `avnlp/self_biorag_processed` and publishing the trained model as `avnlp/BioThink-Qwen3-1.7B`.
- Systematically assessed generations across seven metrics: XML structure integrity, correct Utility/Relevance/Groundness tokens, Answer Correctness (custom GEval), Faithfulness, and Answer Relevancy via DeepEval's LLM-as-a-Judge metrics.

## RAG Model Training

**GitHub:** [https://github.com/avnlp/rag-model-training](https://github.com/avnlp/rag-model-training)  
**Documentation:** [https://deepwiki.com/avnlp/rag-model-training](https://deepwiki.com/avnlp/rag-model-training)  

- Reproduced the original training methodologies behind six advanced RAG techniques, each targeting a different component of the pipeline - retrieval-strategy selection, retrieval-quality evaluation, query refinement, self-reflection, or autonomous agentic search.
- **Adaptive-RAG (SFT):** trained a T5-large query-complexity classifier that routes each query to no-retrieval, single-step, or multi-step retrieval (Simple/Moderate/Complex), on data drawn from Musique and other multi-hop QA datasets.
- **Corrective RAG (SFT):** trained a T5-large retrieval evaluator that grades retrieved documents as Correct/Ambiguous/Incorrect with a −1 to 1 relevance score, powering the decompose-then-recompose correction strategy, on the CRAG dataset.
- **RQ-RAG (SFT):** fine-tuned Llama-3.2-3B with an expanded vocabulary to refine queries through `[S_Rewritten_Query]`, `[S_Decomposed_Query]`, and `[S_Disambiguated_Query]` control tokens under a tree-based generate → retrieve → refine decoding strategy.
- **Self-RAG (two-phase SFT):** trained a T5-base critic and generator in sequence to emit `[Retrieval]`, `[Relevant]`, `[Grounded]`, and `[Utility:1–5]` reflection tokens, applied to the financial Earnings Calls domain.
- **Agentic RAG (GRPO):** fine-tuned Llama-3.1-8B with LoRA on TriviaQA + FAISS retrieval (vLLM-served, up to 6 search-refinement cycles) using Correctness (LLM-as-a-Judge) and Formatting rewards for autonomous missing-information detection, query rewriting, and tool-call generation.
- **ReZero (GRPO):** fine-tuned Llama-3.2-8B with rank-64 LoRA on TriviaQA + multilingual-e5-large + FAISS (up to 32 cycles), rewarding search persistence through six composite signals - Correctness, Formatting, Retry, EM-Chunk, Search Strategy, and Search Diversity - over `think`/`search`/`answer` tags.

## Group Relative Policy Optimization (GRPO)

**GitHub:** [https://github.com/avnlp/grpo](https://github.com/avnlp/grpo)  
**Documentation:** [https://deepwiki.com/avnlp/grpo](https://deepwiki.com/avnlp/grpo)  

- Aggregated and refactored four open-source GRPO implementations under a shared structure, isolating how each makes different systems choices around the same core algorithm.
- GRPO replaces the learned value function with the group-average reward of sampled responses as the baseline, normalizes rewards within each group into per-output advantages, and regularizes with a KL term in the loss - eliminating the memory and compute overhead of PPO-style critics.
- Unified four implementations under one codebase: **nanoAhaMoment** (vLLM with sleep/wake GPU sharing, DeepSpeed ZeRO-2, REINFORCE+KL), **GRPO:Zero** (custom Transformer/KV-cache loop on Qwen2.5-3B-Instruct, pure REINFORCE, no reference model), **Simple GRPO** (DeepSpeed, reference log-probs offloaded to a separate Bottle HTTP server, PPO-clip+KL), and **GRPO from Scratch** from Andriy Burkov's LM Book (pure PyTorch, `copy.deepcopy` reference per outer iteration, μ updates per batch).
- Dissected the sharpest divergence - the generation backend: vLLM with PagedAttention and sleep mode, a hand-written autoregressive KV-cache loop, and HuggingFace `model.generate()` with captured generation log-probs for exact-policy ratio clipping.
- Contrasted reference-policy handling across the four: a CPU-offloaded DeepSpeed copy, no reference at all, a frozen out-of-process server, and a per-iteration `copy.deepcopy` that lags by one iteration.
- Spanned the Countdown and GSM8K tasks across Qwen2.5 (0.5B–7B) models with combined correctness + format rewards over `<think>/<answer>` (or `<reasoning>/<answer>`) tags, documenting each variant's group size, KL coefficient, clip ε, reward range, and optimizer/learning rate.

## LLM Fine-tuning

**GitHub:** [https://github.com/avnlp/llm-finetuning](https://github.com/avnlp/llm-finetuning)  
**Documentation:** [https://deepwiki.com/avnlp/llm-finetuning](https://deepwiki.com/avnlp/llm-finetuning)  

- Built 39 fine-tuning pipelines across 16 datasets spanning three paradigms - adapter-based SFT, GRPO reinforcement learning, and preference alignment - on **TRL**, **PEFT**, and **Unsloth** with composable reward functions and LLM-as-a-Judge evaluation.
- **Adapter SFT (25 pipelines):** fine-tuned Llama-3.2-3B with five parameter-efficient methods - LoRA and QLoRA (rank 8, alpha 32), DoRA (weight-decomposed), P-Tuning, and Prefix-Tuning - on ARC, TriviaQA, FactScore, PopQA, and Earnings Calls, comparing the adapter techniques head-to-head.
- **Math reasoning (GRPO):** trained Phi-4, Mistral-7B, Llama-3.2-3B, Llama-3.1-8B, and Gemma3-1B on GSM8K with one correctness reward (numeric extraction from `<answer>`) and four format rewards (reasoning-tag nesting, ≥3 numbered steps, multi-line depth, block presence); a two-stage Qwen-3 Base pipeline first ran SFT on OpenR1-Math-220k to prime the format, then GRPO on GSM8K.
- **Multi-hop and medical QA (GRPO):** fine-tuned Llama-3.2-3B with eight reward functions - four correctness (DeepEval GEval-for-RAG, Summarization, Answer Relevancy, Evidently AI CorrectnessLLMEval) and four format - on HotpotQA, FreshQA, and MuSiQue, then reused the same architecture for biomedical reasoning on MedQA, BioASQ, and PubMedQA.
- **Preference alignment:** aligned models over QLoRA with four algorithms - DPO (Zephyr-7B/UltraFeedback, Llama-3-8B/WebGPT), ORPO (Llama-3-8B/UltraFeedback), KTO (Qwen2.5-1.5B/KTO-Mix-14k), and PPO (Llama-3-8B on UltraFeedback and WebGPT, scored by an OpenAssistant DeBERTa-v3 reward model).

## RAG Pipelines

**GitHub:** [https://github.com/avnlp/rag-pipelines](https://github.com/avnlp/rag-pipelines)  
**Documentation:** [https://deepwiki.com/avnlp/rag-pipelines](https://deepwiki.com/avnlp/rag-pipelines)  

- Built advanced, domain-specific RAG pipelines for medical and financial question answering on one standardized **LangGraph** architecture split into offline indexing and online evaluation stages.
- Composed the stack from LangGraph for async orchestration, **BAML** for typed structured generation, **Unstructured** for document processing, **Milvus** for dense + BM25 hybrid retrieval with Reciprocal Rank Fusion, **Contextual AI** instruction-following rerankers, and **DeepEval** with **Confident AI** for evaluation and tracing.
- Engineered a three-layer metadata enrichment system: Structural (rule-based, zero-LLM - content hashes, counts, language, headings), Dynamic (user-defined fields extracted by an LLM per a YAML schema), and Fixed (RAG-optimized fields - candidate questions, summary, keywords, content type), with content-hash caching and three cost/quality modes.
- Wired the query-time pipeline to parse each question into structured metadata and a vector-DB filter expression, run hybrid retrieval, rerank with domain-specific instructions, generate a chain-of-thought answer via a typed BAML function, and score with contextual recall, contextual precision, contextual relevancy, answer relevancy, and faithfulness.
- Defined every LLM call as a typed BAML function with Schema-Aligned Parsing (recovers malformed JSON and missing fields, no tool-calling APIs required), per-domain prompt templates, and a Groq → Cerebras → SambaNova multi-provider fallback chain with exponential-backoff retries.
- Targeted six datasets: **HealthBench**, **MedCaseReasoning**, **MetaMedQA**, and **PubMedQA** (medical), and **FinanceBench** and **Earnings Calls** (financial).

## Advanced RAG Pipeline Optimization with DSPy

**GitHub:** [https://github.com/avnlp/dspy-opt](https://github.com/avnlp/dspy-opt)  
**Documentation:** [https://deepwiki.com/avnlp/dspy-opt](https://deepwiki.com/avnlp/dspy-opt)  

- Optimized modular DSPy RAG pipelines by automatically tuning prompt instructions and few-shot demonstrations against DeepEval metrics, eliminating hand-crafted prompts.
- Assembled each pipeline from composable modules - a `QueryRewriter` (synonym expansion and noise removal), a `SubQueryGenerator` (decomposition into parallel sub-queries), a `MetadataExtractor` (LLM-parsed JSON schema for filtering), a `WeaviateRetriever` (hybrid vector + keyword search), and a `dspy.ChainOfThought` answer generator - with Confident AI logging.
- Integrated five optimizers, distinguished by what they tune and how: **MIPROv2** (joint instructions + demos via Bayesian optimization), **COPRO** (instruction-only coordinate ascent), **BootstrapFewShotWithRandomSearch** (demo-only random search), **SIMBA** (mini-batch self-reflective rule generation), and **GEPA** (reflection-driven genetic evolution over a Pareto frontier with candidate merging).
- Drove optimization and evaluation with DeepEval's Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall, and Contextual Relevancy.
- Evaluated across five datasets spanning complexity types - FreshQA/SealQA (single-hop, false-premise debunking), HotpotQA (multi-hop), PubMedQA (biomedical), TriviaQA (factoid), and Wikipedia with WikiQA pairs (general) - all configured through YAML for models, retrievers, and optimizer hyperparameters.

## VectorDB

**GitHub:** [https://github.com/avnlp/vectordb](https://github.com/avnlp/vectordb)  
**Documentation:** [https://deepwiki.com/avnlp/vectordb](https://deepwiki.com/avnlp/vectordb)  

- Built a unified, production-oriented toolkit for semantic search and RAG across five vector databases - **Pinecone**, **Weaviate**, **Milvus**, **Qdrant**, and **Chroma** - with feature parity between Haystack and LangChain and a single configuration-driven benchmarking surface.
- Implemented three retrieval modes: dense search (any SentenceTransformers model), sparse search (SPLADE or native BM25 for exact terminology), and hybrid search fusing both via RRF or weighted scoring, with metadata filters and optional Groq/OpenAI answer generation.
- Engineered advanced RAG patterns: cross-encoder and API reranking, MMR and diversity filtering, query enhancement (multi-query, HyDE, step-back), contextual compression, parent-document retrieval, JSON indexing, and an agentic RAG loop that searches, reflects, and refines.
- Implemented namespaces and multi-tenancy per backend (Pinecone namespaces, Milvus partition keys, collection-based separation for Qdrant, Chroma, and Weaviate), alongside cost-optimized RAG using local sparse embeddings.
- Benchmarked across TriviaQA, ARC, PopQA, FactScore, and Earnings Calls with dataset loaders and standardized Recall@k, Precision@k, MRR, NDCG@k, and hit-rate metrics for consistent comparison across backends.

## LLM Rankers

**GitHub:** [https://github.com/avnlp/rankers](https://github.com/avnlp/rankers)  
**Documentation:** [https://deepwiki.com/avnlp/rankers](https://deepwiki.com/avnlp/rankers)  
**Paper:** [LLM Rankers](https://github.com/avnlp/rankers/blob/main/paper/rankers.pdf)  

- Implemented Pairwise, Setwise, and Listwise LLM ranking components for Haystack, using structured generation and Pydantic validation that keep zero-shot ranking well-formed even on smaller models.
- Accelerated Pairwise and Setwise ranking with efficient sorting (Heapsort and Bubblesort) - all-pairs and heapsort for Pairwise, multi-document heapsort and bubblesort-style strategies for Setwise - building the core sorting on `ielab/llm-rankers`.
- Integrated the **RankLLM** framework for Listwise ranking with sliding-window reranking and ranking-specialized models (RankGPT, RankLlama, RankVicuna, RankZephyr).
- Released a custom Evaluator and `ir_datasets` dataloader reporting NDCG, MAP, Recall, and Precision at multiple cutoffs on FIQA, SciFact, NFCorpus, TREC-DL 2019, and TREC-DL 2020 with Mistral, Phi-3, and Llama-3 base models.
- Found all rankers performed closely; RankZephyr and RankLlama (with the Listwise ranker) edged out the rest, while among general-purpose LLMs, Llama-3 with the Setwise and Pairwise rankers performed best.

**Results (NDCG@10):**

| Model | Ranker | FiQA | SciFact | NFCorpus | TREC-19 | TREC-20 |
|---|---|---|---|---|---|---|
| Instructor-XL (dense baseline) | — | 0.4650 | 0.6920 | 0.4180 | 0.5230 | 0.5040 |
| Mistral | Setwise · Heapsort | 0.4680 | 0.6960 | 0.4310 | 0.7140 | 0.6950 |
| Phi-3 | Setwise · Heapsort | 0.4710 | 0.7120 | 0.4390 | 0.7220 | 0.7030 |
| Llama-3 | Setwise · Heapsort | 0.4760 | 0.7760 | 0.4430 | 0.7460 | 0.7270 |
| RankLlama | Listwise | 0.4796 | 0.7812 | 0.4518 | 0.7511 | 0.7642 |
| **RankZephyr** | **Listwise** | **0.4892** | **0.7891** | **0.4578** | **0.7693** | **0.7743** |

## Pairwise Ranking Prompting (PRP)

**GitHub:** [https://github.com/avnlp/prp](https://github.com/avnlp/prp)  
**Documentation:** [https://deepwiki.com/avnlp/prp](https://deepwiki.com/avnlp/prp)  

- Implemented Pairwise Ranking Prompting ([Qin et al., 2023](https://arxiv.org/abs/2306.17563)) as a standalone, zero-shot LLM reranking library that compares documents in pairs instead of scoring them individually, mitigating LLM position bias through bidirectional (A vs. B and B vs. A) comparisons.
- Built three sorting strategies: `all_pair` (enumerate all pairs and aggregate by win ratio, O(N²) calls - order-insensitive), `heapsort` (LLM as comparator at O(N log N), largely order-insensitive), and `sliding_k` (bottom-up sliding window with stride 1 at O(K·N) - favorable complexity but order-dependent).
- Engineered support for any OpenAI-compatible API (OpenAI, Groq, local models) with structured generation, Pydantic validation, and a Haystack-based evaluation toolkit over `ir_datasets` (BM25 retrieves top-100 candidates, then PRP reranks).
- Evaluated on FIQA, SciFact, NFCorpus, TREC-DL 2019, and TREC-DL 2020 with NDCG, MAP, Recall, and Precision across Mistral, Phi-3, and Llama-3.
- Found Llama-3 with `all_pair` strongest across every dataset on NDCG@10, with `heapsort` the recommended quality/efficiency tradeoff.

**Results (NDCG@10):**

| Model | Method | FiQA | SciFact | NFCorpus | TREC-19 | TREC-20 |
|---|---|---|---|---|---|---|
| Mistral | PRP-allpair | 0.4676 | 0.6860 | 0.4312 | 0.7186 | 0.6987 |
| Phi-3 | PRP-allpair | 0.4714 | 0.7028 | 0.4386 | 0.7228 | 0.7167 |
| Llama-3 | PRP-heapsort | 0.4764 | 0.7765 | 0.4423 | 0.7508 | 0.7637 |
| Llama-3 | PRP-sliding_k | 0.4793 | 0.7852 | 0.4503 | 0.7511 | 0.7642 |
| **Llama-3** | **PRP-allpair** | **0.4992** | **0.7912** | **0.4658** | **0.7623** | **0.7671** |

## LLM Blender

**GitHub:** [https://github.com/avnlp/llm-blender](https://github.com/avnlp/llm-blender)  
**Documentation:** [https://deepwiki.com/avnlp/llm-blender](https://deepwiki.com/avnlp/llm-blender)  
**Paper:** [LLM Ensembling: Haystack Pipelines with LLM-Blender](https://github.com/avnlp/llm-blender/blob/main/paper/llm_blender.pdf)  

- Integrated LLM-Blender with Haystack RAG pipelines to ensemble the outputs of multiple LLMs into a single, higher-quality answer.
- Implemented LLM-Blender's two-stage design: a **PairRanker** ranks candidate generations through pairwise comparison (encoding input and candidates with a RoBERTa cross-attention encoder to build a pairwise-comparison matrix), then a **GenFuser** seq2seq-fuses the top-K candidates conditioned on the input.
- Built a custom `LLMBlenderRanker` Haystack component wrapping PairRanker, comparing each candidate against the input pairwise so subtle wording differences between models don't skew the ensemble - unlike MLM-scoring, SimCLS, or SummaReranker, which score candidates individually.
- Ensembled Mistral-7B, Llama-3-8B, Phi-3-mini, OpenChat-3.5, Starling-LM-7B-alpha, and OpenHermes-2.5, benchmarking on MixInstruct and BillSum with BERTScore, BARTScore, and BLEURT.
- Found that PairRanker ensembles of these newer open models outperform their individual outputs and exceed the original LLM-Blender authors' reported baselines on MixInstruct across all three metrics.

**Results (MixInstruct, PairRanker ensemble of Llama-3-8B + Phi-3-mini + Mistral-7B):**

| Configuration | BERTScore ↑ | BARTScore ↑ | BLEURT ↑ |
|---|---|---|---|
| LLM-Blender authors (PairRanker) | 72.97 | −3.14 | −0.37 |
| **This work (PairRanker)** | **75.83** | **−2.87** | **−0.26** |

## RRF: Performance Evaluation of Rankers and RRF Techniques for Retrieval Pipelines

**GitHub:** [https://github.com/avnlp/rrf](https://github.com/avnlp/rrf)  
**Documentation:** [https://deepwiki.com/avnlp/rrf](https://deepwiki.com/avnlp/rrf)  
**Paper:** [Performance Evaluation of Rankers and RRF Techniques for Retrieval Pipelines](https://github.com/avnlp/rrf/blob/main/paper/rankers_rrf.pdf)  

- Evaluated rankers and Reciprocal Rank Fusion for LFQA/RAG pipelines, where context-window ordering and redundancy directly shape answer quality.
- Implemented three rankers: a **Diversity Ranker** (maximizes document diversity via embedding similarity), a **Lost-in-the-Middle Ranker** (places the most relevant documents at the start and end of the prompt to counter the lost-in-the-middle effect), and a **Transformers Similarity Ranker** (cross-encoder query-document scoring).
- Studied dense retrieval with INSTRUCTOR-XL and all-mpnet-base-v2 against hybrid retrieval that fuses BM25 sparse search with dense results via RRF, using bge-reranker-large (Similarity) and ms-marco-MiniLM-L-12-v2 (Diversity).
- Benchmarked every retriever-and-ranker combination on the FIQA dataset using NDCG, MAP, Recall, and Precision.
- Found the best results with INSTRUCTOR-XL and the Similarity Ranker, then Diversity, then Lost-in-the-Middle - and that instruction-tuned embeddings like INSTRUCTOR-XL outperform many ranker combinations by exploiting data-specific instructions.

## Effect of Optimizer Selection and Hyperparameter Tuning on Training Efficiency and LLM Performance

**GitHub:** [https://github.com/avnlp/hyperparameter-tuning](https://github.com/avnlp/hyperparameter-tuning)  
**Documentation:** [https://deepwiki.com/avnlp/hyperparameter-tuning](https://deepwiki.com/avnlp/hyperparameter-tuning)  
**Paper:** [Effect of Optimizer Selection and Hyperparameter Tuning on Training Efficiency and LLM Performance](https://github.com/avnlp/hyperparameter-tuning/blob/main/paper/optimizer_inclusions.pdf)  

- Demonstrated how optimizer choice and the hyperparameter-tuning protocol jointly determine training efficiency and final model quality, with optimizer rankings highly sensitive to how thoroughly each optimizer is tuned.
- Mapped the optimizer inclusion relationships - `SGD ⊆ Momentum ⊆ RMSProp`, `SGD ⊆ Momentum ⊆ Adam`, and `SGD ⊆ Nesterov ⊆ Nadam` - across SGD, Momentum, NAG, RMSProp, and Adam/AdamW, tuning learning rate, momentum, smoothing constant, and the β1/β2 moment-estimate decay rates.
- Conducted experiments across three NLP tasks: Sentiment Analysis (Financial PhraseBank, StockTwits, FinGPT-Sentiment), Question Answering (SQuAD, CoQA, FIQA), and Summarization (Multi-News, BillSum).
- Fine-tuned DistilBERT, BERT, FinBERT, RoBERTa, Llama-3, and Phi-3 for classification and QA, and BART, DistilBART, and T5 for summarization.
- Found that with sufficient tuning a more expressive optimizer never underperforms its special cases - RMSProp and AdamW never lost to SGD, Momentum, or Nesterov - confirming that inclusion relationships predict optimizer performance as tuning approaches optimality.
