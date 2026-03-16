"""
FinBERT Sentiment Analyzer for earnings call transcripts.
Uses yiyanghkust/finbert-tone for financial sentiment classification.
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-load heavy dependencies
_PIPELINE = None
_LOAD_ERROR: Optional[str] = None
_AVAILABLE = False


def _try_load_pipeline():
    """Lazy-load FinBERT pipeline on first use."""
    global _PIPELINE, _LOAD_ERROR, _AVAILABLE
    if _PIPELINE is not None:
        return _PIPELINE

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)

        model_name = "yiyanghkust/finbert-tone"
        logger.info("Loading FinBERT model: %s", model_name)
        
        # Load with caching enabled
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,  # Try local first
            cache_dir=".cache/finbert"  # Use local cache
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=True,
            cache_dir=".cache/finbert"
        )
        
        device = 0 if torch.cuda.is_available() else -1

        _PIPELINE = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            framework="pt",
            device=device,
            return_all_scores=True,
            top_k=None  # Silence deprecation warning
        )
        _AVAILABLE = True
        logger.info("FinBERT loaded successfully (device=%s)", "cuda" if device == 0 else "cpu")
        return _PIPELINE
    except Exception as e:
        try:
            # If local load failed, try downloading
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=".cache/finbert"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=".cache/finbert"
            )
            device = 0 if torch.cuda.is_available() else -1

            _PIPELINE = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                framework="pt",
                device=device,
                return_all_scores=True,
                top_k=None
            )
            _AVAILABLE = True
            logger.info("FinBERT downloaded and loaded successfully (device=%s)", "cuda" if device == 0 else "cpu")
            return _PIPELINE
        except Exception as e2:
            _LOAD_ERROR = str(e2)
            _AVAILABLE = False
            logger.warning("FinBERT unavailable: %s", e2)
            return None


@dataclass
class FinBERTResult:
    polarity_mean: float = 0.0
    polarity_std: float = 0.0
    pos_ratio: float = 0.0
    neg_ratio: float = 0.0
    neu_ratio: float = 0.0
    sentence_scores: List[Dict] = field(default_factory=list)
    top_positive: List[Dict] = field(default_factory=list)
    top_negative: List[Dict] = field(default_factory=list)
    probability: float = 0.5
    available: bool = False
    error: Optional[str] = None


class FinBERTAnalyzer:
    """FinBERT-based financial sentiment analyzer with batch processing."""

    def __init__(self, max_sentences: int = 200, batch_size: int = 16):
        self.max_sentences = max_sentences
        self.batch_size = batch_size

    @staticmethod
    def is_available() -> bool:
        pipe = _try_load_pipeline()
        return pipe is not None

    def analyze(self, sentences: List[str], top_k: int = 5) -> FinBERTResult:
        pipe = _try_load_pipeline()

        if pipe is None:
            return FinBERTResult(
                available=False,
                error=_LOAD_ERROR or "FinBERT not loaded",
            )

        sents = sentences[: self.max_sentences]
        if not sents:
            return FinBERTResult(available=True)

        all_preds = []
        try:
            for i in range(0, len(sents), self.batch_size):
                batch = sents[i : i + self.batch_size]
                outputs = pipe(batch, truncation=True, max_length=256)
                all_preds.extend(outputs)
        except Exception as e:
            logger.exception("FinBERT inference error: %s", e)
            return FinBERTResult(available=True, error=str(e))

        scores = []
        for sent, pred in zip(sents, all_preds):
            label_scores = {
                entry.get("label", "").lower(): float(entry.get("score", 0.0))
                for entry in pred
            }
            p_pos = label_scores.get("positive", 0.0)
            p_neg = label_scores.get("negative", 0.0)
            p_neu = label_scores.get("neutral", 0.0)
            polarity = p_pos - p_neg

            scores.append({
                "sentence": sent[:200],
                "polarity": polarity,
                "positive": p_pos,
                "negative": p_neg,
                "neutral": p_neu,
            })

        polarities = np.array([s["polarity"] for s in scores])
        polarity_mean = float(polarities.mean())
        polarity_std = float(polarities.std()) if len(polarities) > 1 else 0.0

        pos_ratio = float((polarities > 0.05).mean())
        neg_ratio = float((polarities < -0.05).mean())
        neu_ratio = max(0.0, 1.0 - pos_ratio - neg_ratio)

        # Map polarity from [-1, 1] to probability [0, 1]
        probability = float(np.clip((polarity_mean + 1.0) / 2.0, 0.0, 1.0))

        sorted_by_polarity = sorted(scores, key=lambda x: x["polarity"], reverse=True)
        top_positive = sorted_by_polarity[:top_k]
        top_negative = sorted_by_polarity[-top_k:][::-1]

        return FinBERTResult(
            polarity_mean=polarity_mean,
            polarity_std=polarity_std,
            pos_ratio=pos_ratio,
            neg_ratio=neg_ratio,
            neu_ratio=neu_ratio,
            sentence_scores=scores,
            top_positive=top_positive,
            top_negative=top_negative,
            probability=probability,
            available=True,
        )
