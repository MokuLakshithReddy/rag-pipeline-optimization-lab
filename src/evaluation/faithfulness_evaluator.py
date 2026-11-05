"""Dedicated Natural Language Inference (NLI) and factual entailment evaluator for RAG.

Evaluates whether generated answer claims are factually entailed, neutral, or
contradicted by retrieved context using dedicated 3-class NLI models or LLM-as-a-judge.
"""

from __future__ import annotations
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class FactualEntailmentEvaluator:
    """Evaluates factual faithfulness using genuine 3-class NLI models or LLM verification."""

    def __init__(
        self,
        nli_model_name: Optional[str] = None,
        device: str = "cpu",
        lazy_load: bool = True,
        llm_judge: Optional[Any] = None
    ):
        # Default to environment or None (fast deterministic heuristic NLI)
        self.nli_model_name = nli_model_name or os.getenv("NLI_MODEL_NAME")
        self.device = device
        self.llm_judge = llm_judge
        self._nli_model = None
        self._load_failed = False
        self._load_error: Optional[str] = None

        if self.nli_model_name and not lazy_load:
            self._ensure_model()

    @property
    def active_evaluator_mode(self) -> str:
        """Return the active evaluation mode and model name."""
        self._ensure_model()
        if self._nli_model is not None and not self._load_failed:
            return f"transformer_nli ({self.nli_model_name})"
        elif self.llm_judge is not None:
            model_id = getattr(self.llm_judge, "model_name", "custom_llm")
            return f"llm_judge ({model_id})"
        return "heuristic_claim_verifier"

    def _ensure_model(self) -> None:
        """Lazy load dedicated 3-class NLI Cross-Encoder if a model name is configured."""
        if not self.nli_model_name:
            return

        if self._nli_model is None and not self._load_failed:
            try:
                from sentence_transformers import CrossEncoder
                self._nli_model = CrossEncoder(self.nli_model_name, device=self.device)
                logger.info(f"Loaded dedicated 3-class NLI model: {self.nli_model_name}")
            except Exception as e:
                self._load_failed = True
                self._load_error = str(e)
                logger.warning(
                    f"Could not load NLI model '{self.nli_model_name}': {e}. "
                    f"Falling back to heuristic claim entailment verification."
                )

    def extract_claims(self, text: str) -> List[str]:
        """Extract atomic factual claims and assertions from generated text."""
        if not text or not text.strip():
            return []
        raw_sentences = re.split(r'(?<=[.!?])\s+|\n+|- ', text.strip())
        claims: List[str] = []
        for s in raw_sentences:
            clean = re.sub(r'\[Source:[^\]]+\]', '', s).strip()
            # Exclude short boilerplate or conversational prefixes
            if len(clean) > 8 and not clean.lower().startswith("based on the provided"):
                claims.append(clean)
        return claims or [text.strip()]

    def segment_context_sentences(self, context: str) -> List[str]:
        """Decompose context passages into discrete premise sentences."""
        if not context or not context.strip():
            return []
        sentences = [
            s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', context)
            if len(s.strip()) > 15
        ]
        return sentences or [context.strip()]

    def evaluate_claim_nli(self, claim: str, context_sentences: List[str]) -> Tuple[str, float]:
        """Evaluate a claim against context premises using dedicated 3-class NLI inference.
        
        Returns:
            (verdict, score) where verdict is 'entailed', 'contradicted', or 'neutral'.
        """
        self._ensure_model()

        if self._nli_model is not None and not self._load_failed:
            try:
                # Prepare premise-hypothesis pairs: (premise=context_sentence, hypothesis=claim)
                candidate_premises = context_sentences[:12]
                pairs = [[premise, claim] for premise in candidate_premises]
                logits = self._nli_model.predict(pairs, show_progress_bar=False)

                # Softmax over logits: [contradiction, entailment, neutral]
                exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

                # Map labels according to model config or standard NLI ordering
                # Usually: 0: contradiction, 1: entailment, 2: neutral (or check id2label)
                id2label = getattr(self._nli_model.model.config, "id2label", {
                    0: "contradiction", 1: "entailment", 2: "neutral"
                })
                # Normalize key names
                label_indices = {}
                for idx, lbl in id2label.items():
                    lbl_lower = lbl.lower()
                    if "entail" in lbl_lower:
                        label_indices["entailment"] = int(idx)
                    elif "contra" in lbl_lower:
                        label_indices["contradiction"] = int(idx)
                    else:
                        label_indices["neutral"] = int(idx)

                entail_idx = label_indices.get("entailment", 1)
                contra_idx = label_indices.get("contradiction", 0)

                max_entail_prob = float(np.max(probs[:, entail_idx]))
                max_contra_prob = float(np.max(probs[:, contra_idx]))

                if max_contra_prob > 0.6:
                    return "contradicted", 0.0
                elif max_entail_prob > 0.5:
                    return "entailed", max_entail_prob
                else:
                    return "neutral", max_entail_prob
            except Exception as e:
                logger.warning(f"NLI model inference error: {e}, falling back to heuristic verification.")

        # 2. LLM-as-a-Judge Evaluation if configured
        if self.llm_judge is not None:
            return self._evaluate_claim_llm_judge(claim, context_sentences)

        # 3. Heuristic Anchor & Relational Entailment
        return self._heuristic_claim_verification(claim, context_sentences)

    def _evaluate_claim_llm_judge(self, claim: str, context_sentences: List[str]) -> Tuple[str, float]:
        """Invoke configured LLM judge for proposition entailment verification."""
        try:
            premise_text = " ".join(context_sentences[:8])
            prompt = (
                f"Premise: {premise_text}\n"
                f"Hypothesis: {claim}\n"
                f"Does the Premise entail the Hypothesis? Answer ONLY with ENTAILED, CONTRADICTED, or NEUTRAL:"
            )
            resp = self.llm_judge.generate(
                query=prompt,
                context=premise_text,
                citations=[],
                retrieved_chunks=[]
            )
            raw = (resp.answer or "").upper()
            if "CONTRADICT" in raw:
                return "contradicted", 0.0
            elif "ENTAIL" in raw:
                return "entailed", 1.0
            else:
                return "neutral", 0.5
        except Exception as e:
            logger.warning(f"LLM judge inference error: {e}, falling back to heuristic verification.")
            return self._heuristic_claim_verification(claim, context_sentences)

    def _heuristic_claim_verification(self, claim: str, context_sentences: List[str]) -> Tuple[str, float]:
        """Precise entity-grounded heuristic entailment verification."""
        context_full = " ".join(context_sentences)
        context_lower = context_full.lower()
        claim_lower = claim.lower()

        # 1. Numerical & Acronym Grounding
        numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', claim))
        acronyms = set(re.findall(r'\b[A-Z]{2,}\b', claim))

        for num in numbers:
            if num not in context_full:
                return "contradicted", 0.0
        for acr in acronyms:
            if acr not in context_full and acr.lower() not in context_lower:
                return "contradicted", 0.0

        # 2. Relational Coverage
        claim_tokens = set(re.findall(r'\b\w{3,}\b', claim_lower))
        stopwords = {
            "the", "and", "that", "this", "with", "from", "for", "are", "was", "were",
            "been", "have", "has", "had", "which", "when", "where", "into", "their"
        }
        informative = claim_tokens - stopwords
        if not informative:
            return "entailed", 1.0

        negation_words = {"not", "never", "cannot", "no", "neither", "fails", "without"}
        claim_negated = bool(informative & negation_words)

        best_score = 0.0
        for s in context_sentences:
            s_lower = s.lower()
            s_tokens = set(re.findall(r'\b\w{3,}\b', s_lower))
            coverage = len(informative & s_tokens) / len(informative)
            s_negated = bool(s_tokens & negation_words)
            if claim_negated != s_negated and coverage > 0.4:
                coverage *= 0.2

            if coverage > best_score:
                best_score = coverage

        if best_score >= 0.6:
            return "entailed", float(best_score)
        elif best_score >= 0.3:
            return "neutral", float(best_score)
        else:
            return "unsupported", float(best_score)

    def evaluate_faithfulness(self, prediction: str, context: str) -> Dict[str, Any]:
        """Compute comprehensive factual faithfulness metrics for an answer against context.
        
        Returns:
            Dict containing:
            - faithfulness_score: float in [0.0, 1.0]
            - total_claims: int
            - entailed_claims: int
            - contradicted_claims: int
            - neutral_claims: int
            - claims_details: List of per-claim evaluation results
        """
        if not context or not context.strip():
            return {
                "faithfulness_score": 0.0,
                "total_claims": 0,
                "entailed_claims": 0,
                "contradicted_claims": 0,
                "neutral_claims": 0,
                "claims_details": []
            }

        claims = self.extract_claims(prediction)
        if not claims:
            return {
                "faithfulness_score": 1.0,
                "total_claims": 0,
                "entailed_claims": 0,
                "contradicted_claims": 0,
                "neutral_claims": 0,
                "claims_details": []
            }

        context_sentences = self.segment_context_sentences(context)
        claim_details = []
        entailed_count = 0
        contra_count = 0
        neutral_count = 0

        for claim in claims:
            verdict, score = self.evaluate_claim_nli(claim, context_sentences)
            if verdict == "entailed":
                entailed_count += 1
            elif verdict == "contradicted":
                contra_count += 1
            else:
                neutral_count += 1

            claim_details.append({
                "claim": claim,
                "verdict": verdict,
                "score": round(score, 4)
            })

        # Calculate faithfulness: entailed proportion, penalized if direct contradictions occur
        faithfulness = (entailed_count / len(claims))
        if contra_count > 0:
            faithfulness = max(0.0, faithfulness - (contra_count / len(claims)) * 0.5)

        return {
            "faithfulness_score": float(min(1.0, max(0.0, faithfulness))),
            "total_claims": len(claims),
            "entailed_claims": entailed_count,
            "contradicted_claims": contra_count,
            "neutral_claims": neutral_count,
            "evaluator_mode": self.active_evaluator_mode,
            "claims_details": claim_details
        }
