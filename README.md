# Personalized Q\&A Search Engine with Neural Re-ranking

## Overview

This project implements a personalized search engine designed to retrieve relevant answers from community Question Answering (Q\&A) datasets. It combines traditional retrieval models (BM25) with advanced techniques including query expansion using Large Language Models (LLMs), neural re-ranking with cross-encoders, and personalization based on user profiles and tags.

The integration of these methods aims to improve the relevance and ranking of search results by tailoring them to the individual user’s interests and historical activity.

---

## Features

* **Traditional Retrieval:** Uses BM25 as a baseline model for initial document retrieval.
* **Query Expansion:** Employs the flan-t5-base LLM to enrich user queries with additional context and intent.
* **Neural Re-ranking:** Applies a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-rank retrieved documents based on relevance.
* **Personalization:** Incorporates user profile keywords and tags to personalize search results, enhancing relevance for individual users.
* **Score Integration:** Combines BM25 scores, neural re-ranking scores, and personalization scores with weighted fusion for optimal ranking.
* **Evaluation:** Evaluated with standard IR metrics such as Precision\@10, Mean Average Precision (MAP), and Normalized Discounted Cumulative Gain (nDCG).

---

## Project Structure

* `data_preprocessing/` – Scripts and utilities for cleaning and preparing the dataset, extracting user features.
* `indexing/` – Code for indexing the dataset with PyTerrier and performing BM25 retrieval.
* `query_expansion/` – Modules for expanding user queries using the flan-t5-base model.
* `reranking/` – Neural re-ranking implementation using cross-encoder models.
* `personalization/` – Logic to extract user preferences and compute personalization scores.
* `score_combination/` – Scripts to combine multiple scores and generate final rankings.
* `evaluation/` – Code to evaluate the system using MAP, nDCG, and Precision metrics.
* `notebooks/` – Jupyter notebooks for experiments and analysis.

---

## Getting Started

### Requirements

* Python 3.8+
* PyTerrier
* Transformers (Hugging Face)
* PyTorch
* scikit-learn
* pandas, numpy

Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. **Data Preprocessing**
   Prepare the dataset by cleaning and extracting user features:

   ```bash
   python data_preprocessing/preprocess.py --input raw_data.json --output processed_data.json
   ```

2. **Indexing and Retrieval**
   Build index and retrieve initial documents with BM25:

   ```bash
   python indexing/build_index.py --data processed_data.json
   python indexing/retrieve.py --query "user query"
   ```

3. **Query Expansion**
   Expand queries using the flan-t5-base model:

   ```bash
   python query_expansion/expand_query.py --query "user query"
   ```

4. **Neural Re-ranking**
   Re-rank retrieved documents with cross-encoder:

   ```bash
   python reranking/rerank.py --query "expanded query" --documents docs.json
   ```

5. **Personalization and Score Combination**
   Compute personalization scores and combine with retrieval scores:

   ```bash
   python personalization/compute_scores.py --user_profile user.json --documents docs.json
   python score_combination/combine_scores.py --bm25_scores bm25.json --rerank_scores rerank.json --personalization_scores pers.json
   ```

6. **Evaluation**
   Evaluate the final ranked list:

   ```bash
   python evaluation/evaluate.py --results final_ranked.json --ground_truth ground_truth.json
   ```

---

## Results

| Model                                      | MAP          | nDCG         | Precision\@10 |
| ------------------------------------------ | ------------ | ------------ | ------------- |
| BM25 Baseline                              | 0.614        | 0.681        | -             |
| BM25 + Neural Re-ranking + Personalization | **Improved** | **Improved** | **Improved**  |

* Query expansion with LLMs added valuable context.
* Personalization improved relevance for users with specific interests.
* Neural re-ranking refined the results, boosting overall performance.

---

## Future Work

* Implement advanced personalization methods such as collaborative filtering or deep learning recommender systems.
* Scale to larger datasets to test system robustness and efficiency.
* Incorporate real-time personalization based on live user interactions.
* Add user feedback loops to dynamically improve ranking models.

---

## References

* PyTerrier: [https://pyterrier.org/](https://pyterrier.org/)
* Hugging Face Transformers: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
* BM25 Retrieval: Robertson & Zaragoza (2009)
* Cross-encoder/ms-marco-MiniLM-L-6-v2: [https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
* flan-t5-base: [https://huggingface.co/google/flan-t5-base](https://huggingface.co/google/flan-t5-base)

---
