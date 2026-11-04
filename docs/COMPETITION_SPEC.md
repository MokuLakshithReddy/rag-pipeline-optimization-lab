# Kaggle Competition Specification & Audit Report

## Target Competition Overview
- **Competition Name**: Gen AI Project 4 – Building & Optimizing RAG Pipelines
- **Official URL**: `https://www.kaggle.com/competitions/gen-ai-project-4-building-optimizing-rag-pipelines`
- **Competition Slug**: `gen-ai-project-4-building-optimizing-rag-pipelines`
- **Audit Timestamp**: 2026-11-04T18:30:00Z
- **Auditor Role**: Principal AI Engineer & Retrieval Systems Researcher

---

## 1. Audit Findings & Access Status

### 1.1 Direct Endpoint Inspection
- **HTTP Status**: `404 Not Found` (Public endpoint returns HTTP 404 when unauthenticated).
- **Inference**: The competition may be:
  1. A private, community, or classroom competition (e.g. hosted under an educational cohort such as NxtWave GenAI Intensive) accessible only to registered participants.
  2. Gated behind a rules acceptance check that requires authentication.
  3. Hosted under a slightly modified URL or workspace link.

### 1.2 Kaggle CLI Authentication Audit
- **Status**: Unauthenticated.
- **CLI Output**: `Authentication required to call the Kaggle API.`
- **Action Required from User**: To allow automated downloading of competition assets once accessible, configure Kaggle credentials safely.

---

## 2. Secure Kaggle Authentication Instructions

> **CRITICAL SECURITY NOTICE**: Do NOT paste your Kaggle API key or credentials into the chat or commit them into version control.

To configure your Kaggle CLI locally:
1. Log in to your Kaggle account at [kaggle.com](https://www.kaggle.com).
2. Go to your Account Settings: [https://www.kaggle.com/settings/api](https://www.kaggle.com/settings/api).
3. Under the **API** section, click **Create New Token** (or **Generate New Token**). A file named `kaggle.json` will be downloaded.
4. Place `kaggle.json` in your local home directory:
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - **Linux / macOS**: `~/.kaggle/kaggle.json`
   Ensure permissions are restricted (`chmod 600 ~/.kaggle/kaggle.json` on Unix).
5. Alternatively, export environment variables in your local terminal:
   ```powershell
   $env:KAGGLE_USERNAME="<your-kaggle-username>"
   $env:KAGGLE_KEY="<your-kaggle-api-key>"
   ```
6. **Accept Competition Rules**: Open the competition link in your browser while logged in and click **I Understand and Accept** on the Rules tab.

---

## 3. Unresolved Requirements Matrix

Until the official competition dataset and submission portal are accessible, the following requirements remain unverified:

| Requirement Category | Current Status | Default / Fallback Engineering Assumption |
| :--- | :--- | :--- |
| **Dataset Files & Format** | Unresolved | Multi-format support: PDF, Markdown, Plain Text (`.txt`), and JSON/CSV Q&A pairs. |
| **Ground Truth Annotations** | Unresolved | Support both retrieval-only (Queries + Relevant Doc IDs) and generative QA (Queries + Target Answers). |
| **Target Evaluation Metric** | Unresolved | Implement standard retrieval metrics (MRR@K, NDCG@K, Precision@K, Recall@K, Hit@K) and generative metrics (Faithfulness, Context Relevance, Semantic Similarity). |
| **Submission Schema** | Unresolved | Provide schema adapter supporting both: <br>1. `id,prediction` (Answer generation)<br>2. `query_id,retrieved_chunk_ids` (Ranked retrieval). |
| **Internet Access in Notebook** | Unresolved | Notebook is designed for fully offline / self-contained execution with local HuggingFace / sentence-transformers weights or offline FAISS/BM25. |
| **Compute Hardware Limit** | Unresolved | Optimized for CPU and single GPU (CUDA T4 / P100 compatible; low memory footprint < 8GB). |

---

## 4. Engineering Policy & Compliance Statement

In strict adherence to the competition engineering protocol:
1. **No Invented Specifications**: We will not claim compliance with a fictitious submission format or fabricate competition leaderboard scores.
2. **Modular Architecture**: All retrieval, chunking, embedding, reranking, and generation modules are decoupled so they can immediately plug into the exact competition dataset as soon as credentials/access are supplied.
3. **Reproducible Local Evaluation**: A standardized synthetic/curated evaluation benchmark with verified ground truth is included in `data/benchmarks/` to enable rigorous, repeatable metric comparison across all optimization configurations.

