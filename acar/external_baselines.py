from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .data import Example


@dataclass
class BaselineResult:
    method: str
    split: str
    acc: float
    faithfulness: float
    emi: float


class ExternalBaselineRunner:
    """Integration hooks for blueprint baselines requiring separate implementations/tools."""

    def run_acdc(self, split: str, examples: Sequence[Example]) -> BaselineResult:
        # TODO: integrate transformer-circuit/ACDC pipeline and compute causal metrics.
        return BaselineResult(method="ACDC", split=split, acc=float("nan"), faithfulness=float("nan"), emi=float("nan"))

    def run_tokenshap(self, split: str, examples: Sequence[Example]) -> BaselineResult:
        # TODO: integrate TokenSHAP and evaluate primitive alignment (non-causal baseline).
        return BaselineResult(method="TokenSHAP", split=split, acc=float("nan"), faithfulness=float("nan"), emi=float("nan"))

    def run_cot_faithfulness(self, split: str, examples: Sequence[Example]) -> BaselineResult:
        # TODO: integrate CoT faithfulness benchmark path.
        return BaselineResult(method="CoT", split=split, acc=float("nan"), faithfulness=float("nan"), emi=float("nan"))
