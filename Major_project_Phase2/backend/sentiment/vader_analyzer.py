"""
VADER Sentiment Analyzer for earnings call transcripts.
Uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner).
"""
from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
for pkg in ("punkt", "punkt_tab", "vader_lexicon"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"sentiment/{pkg}")
    except (LookupError, OSError):
        nltk.download(pkg, quiet=True)


@dataclass
class VaderResult:
    compound_mean: float = 0.0
    compound_std: float = 0.0
    pos_ratio: float = 0.0
    neg_ratio: float = 0.0
    neu_ratio: float = 0.0
    sentence_scores: List[Dict] = field(default_factory=list)
    top_positive: List[Dict] = field(default_factory=list)
    top_negative: List[Dict] = field(default_factory=list)
    probability: float = 0.5


class VaderAnalyzer:
    """Singleton-style VADER analyzer with sentence-level and aggregate scoring."""

    def __init__(self):
        self._sia = SentimentIntensityAnalyzer()

    @staticmethod
    def _clean_text(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\n\n", ". ").replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _split_sentences(text: str, min_len: int = 8) -> List[str]:
        if not text or len(text.strip()) < min_len:
            return []
        try:
            sents = sent_tokenize(text)
        except Exception:
            sents = re.split(r'(?<=[.!?])\s+', text)
        sents = [s.strip() for s in sents if len(s.strip()) >= min_len]
        return sents if sents else ([text.strip()] if len(text.strip()) >= min_len else [])

    def analyze(self, text: str, top_k: int = 5) -> VaderResult:
        cleaned = self._clean_text(text)
        sentences = self._split_sentences(cleaned)

        if not sentences:
            return VaderResult()

        scores = []
        for sent in sentences:
            vs = self._sia.polarity_scores(sent)
            scores.append({
                "sentence": sent[:200],
                "compound": vs["compound"],
                "pos": vs["pos"],
                "neg": vs["neg"],
                "neu": vs["neu"],
            })

        compounds = np.array([s["compound"] for s in scores])
        compound_mean = float(compounds.mean())
        compound_std = float(compounds.std()) if len(compounds) > 1 else 0.0

        pos_ratio = float((compounds > 0.05).mean())
        neg_ratio = float((compounds < -0.05).mean())
        neu_ratio = max(0.0, 1.0 - pos_ratio - neg_ratio)

        # Map compound_mean from [-1, 1] to probability [0, 1]
        probability = float(np.clip((compound_mean + 1.0) / 2.0, 0.0, 1.0))

        sorted_by_score = sorted(scores, key=lambda x: x["compound"], reverse=True)
        top_positive = sorted_by_score[:top_k]
        top_negative = sorted_by_score[-top_k:][::-1]

        return VaderResult(
            compound_mean=compound_mean,
            compound_std=compound_std,
            pos_ratio=pos_ratio,
            neg_ratio=neg_ratio,
            neu_ratio=neu_ratio,
            sentence_scores=scores,
            top_positive=top_positive,
            top_negative=top_negative,
            probability=probability,
        )
