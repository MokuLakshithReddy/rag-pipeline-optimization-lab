# Source-faithful submission V3

## Purpose

This branch adds a candidate-generation artifact locally and a reusable validator for the original ProctorIQ competition archive. The existing 73.57 Kaggle submission remains the baseline and must not be overwritten.

## Published scoring weights

| Dimension | Weight |
|---|---:|
| Answer accuracy | 25% |
| Groundedness | 25% |
| Retrieval quality | 20% |
| Citation accuracy | 15% |
| Integrity refusal (adversarial subset only) | 15% |

The hidden answer key and exact semantic-similarity implementation are not available in the public archive. Consequently, local validation cannot predict the exact Kaggle score or guarantee a score of 80+.

## Candidate strategy

The source-faithful V3 candidate keeps responses close to the original Markdown evidence, uses exact section headings, and avoids unsupported guarantees, exemptions, timer claims, or support powers. Adversarial questions are answered without helping a candidate bypass assessment integrity controls.

## Validate a candidate

After extracting the competition archive and placing the generated CSV at a local path:

```bash
python scripts/validate_competition_submission.py \
  --archive ./gen-ai-project-4-building-optimizing-rag-pipelines.zip \
  --submission ./submission_source_faithful_v3.csv
```

The validator checks:
- exactly the original 50 question IDs, in test-file order;
- the four required columns;
- non-empty answers and citations;
- one-to-one alignment of pipe-separated document and section citations;
- existence of every cited Markdown file and exact H2 heading.

It does not calculate answer semantic similarity, groundedness semantic similarity, integrity-refusal score, or the hidden Kaggle composite.

## Experiment discipline

Submit V3 as a separate Kaggle experiment. Record the returned score and preserve the 73.57 baseline. Do not replace a higher-scoring submission unless Kaggle confirms an improvement.
