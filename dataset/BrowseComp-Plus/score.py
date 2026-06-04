"""BrowseComp-Plus answer scoring."""

from __future__ import annotations

import re
from typing import Tuple


def _normalize(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n.,;:!?\"'`")
    return s


class BrowseCompScorer:
    def extract_model_answer(self, text: str) -> str:
        return (text or "").strip()

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        gold = _normalize(expected_output)
        pred_raw = self.extract_model_answer(prediction)
        pred = _normalize(pred_raw)
        return (1 if pred == gold else 0), pred_raw
