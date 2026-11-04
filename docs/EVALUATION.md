# Evaluation Methodology & Metrics Reference

## 1. Metric Formulations

### 1.1 Precision@K
$$ \text{Precision@K} = \frac{|\text{Retrieved@K} \cap \text{Relevant}|}{K} $$
Measures the proportion of the top-K retrieved documents that are genuinely relevant.

### 1.2 Recall@K
$$ \text{Recall@K} = \frac{|\text{Retrieved@K} \cap \text{Relevant}|}{|\text{Relevant}|} $$
Measures the coverage of ground-truth relevant documents retrieved in the top-K.

### 1.3 Mean Reciprocal Rank (MRR)
$$ \text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i} $$
Evaluates how early the first relevant document appears in the candidate list.

### 1.4 Normalized Discounted Cumulative Gain (NDCG@K)
$$ \text{DCG@K} = \sum_{r=1}^{K} \frac{\text{rel}_r}{\log_2(r + 1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}} $$
Penalizes relevant documents that appear further down in the top-K results list.

### 1.5 Hit@K
A binary indicator (1.0 or 0.0) reflecting whether at least one relevant passage was found in top-K.

### 1.6 Grounded Faithfulness & Token F1
- **Faithfulness**: Measures the percentage of non-stopword tokens in the synthesized response that are supported directly by the retrieved context.
- **Answer F1**: Token harmonic mean measuring exact lexical alignment with ground-truth reference answers.

