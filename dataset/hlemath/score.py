"""HLEMATH grading (sync), adapted from AFlow benchmarks/hlemath.py (no AFlow dependency)."""

from __future__ import annotations

import re
from math import isclose
from typing import Any, Tuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


class HLEMATHScorer:
    """Extract \\boxed{} answers and compare with sympy / numeric tolerance."""

    def extract_model_answer(self, text: str) -> str:
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        return 0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        if str(prediction) == str(reference):
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                p = self.parse_digits(prediction)
                r = self.parse_digits(reference)
                return isclose(p, r, abs_tol=1e-3)
        except Exception:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except Exception:
            pass

        return False

    def is_digit(self, num: Any) -> bool:
        return self.parse_digits(num) is not None

    def parse_digits(self, num: Any) -> float | None:
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except Exception:
            pass
        if str(num).endswith("%"):
            n = str(num)[:-1]
            if n.endswith("\\"):
                n = n[:-1]
            try:
                return float(n) / 100
            except Exception:
                pass
        return None

    def symbolic_equal(self, a: Any, b: Any) -> bool:
        def _parse(s: Any):
            for f in (parse_latex, parse_expr):
                try:
                    return f(s)
                except Exception:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except Exception:
            pass

        try:
            if isclose(float(N(a)), float(N(b)), abs_tol=1e-3):
                return True
        except Exception:
            pass
        return False


def score_answer(gold_answer: str, model_output: str) -> Tuple[int, str]:
    """Convenience: return (0|1, extracted_prediction)."""
    return HLEMATHScorer().calculate_score(gold_answer, model_output)
