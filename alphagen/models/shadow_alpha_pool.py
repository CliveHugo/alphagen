from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .linear_alpha_pool import MseAlphaPool
from ..data.calculator import AlphaCalculator
from ..data.expression import Expression, OutOfDataRangeError


@dataclass
class ShadowAlphaRecord:
    expr: Expression
    expr_str: str
    single_ic: float
    evicted_weight: float
    evicted_at_eval: int
    reason: str
    seen_count: int = 1
    approx_score: float = 0.
    delta_ic: Optional[float] = None
    suggested_weight: float = 0.
    exact_evaluated: bool = False
    last_error: Optional[str] = None
    active_mutual_ics: Optional[List[float]] = field(default=None, repr=False)

    def to_json_dict(self, rank: int) -> Dict[str, Any]:
        return {
            "rank": rank,
            "tier": "shadow",
            "expr": self.expr_str,
            "delta_ic": None if self.delta_ic is None else float(self.delta_ic),
            "approx_score": float(self.approx_score),
            "suggested_weight": float(self.suggested_weight),
            "single_ic": float(self.single_ic),
            "evicted_weight": float(self.evicted_weight),
            "evicted_at_eval": self.evicted_at_eval,
            "reason": self.reason,
            "seen_count": self.seen_count,
            "exact_evaluated": self.exact_evaluated,
            "last_error": self.last_error,
        }


class ShadowAlphaPool:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.records: List[ShadowAlphaRecord] = []
        self._by_expr: Dict[str, ShadowAlphaRecord] = {}

    def add(
        self,
        expr: Expression,
        single_ic: float,
        evicted_weight: float,
        evicted_at_eval: int,
        reason: str
    ) -> None:
        if self.capacity <= 0:
            return

        expr_str = str(expr)
        record = self._by_expr.get(expr_str)
        if record is not None:
            record.expr = expr
            record.single_ic = single_ic
            record.evicted_weight = evicted_weight
            record.evicted_at_eval = evicted_at_eval
            record.reason = reason
            record.seen_count += 1
            record.approx_score = abs(single_ic)
            record.delta_ic = None
            record.suggested_weight = evicted_weight
            record.exact_evaluated = False
            record.last_error = None
            record.active_mutual_ics = None
            return

        record = ShadowAlphaRecord(
            expr=expr,
            expr_str=expr_str,
            single_ic=single_ic,
            evicted_weight=evicted_weight,
            evicted_at_eval=evicted_at_eval,
            reason=reason,
            approx_score=abs(single_ic),
            suggested_weight=evicted_weight
        )
        self.records.append(record)
        self._by_expr[expr_str] = record
        self.trim()

    def trim(self) -> None:
        if self.capacity <= 0:
            self.records = []
            self._by_expr = {}
            return
        if len(self.records) <= self.capacity:
            return

        self.records.sort(key=self._trim_key, reverse=True)
        self.records = self.records[:self.capacity]
        self._by_expr = {record.expr_str: record for record in self.records}

    def _trim_key(self, record: ShadowAlphaRecord) -> Any:
        if record.delta_ic is not None:
            score = record.delta_ic
        else:
            score = record.approx_score
        return score, record.evicted_at_eval


class ShadowMseAlphaPool(MseAlphaPool):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        device: torch.device = torch.device("cpu"),
        shadow_capacity: int = 200,
        shadow_export_top_k: Optional[int] = None,
        shadow_rank_exact_top_n: Optional[int] = None,
        shadow_refit: bool = False
    ):
        super().__init__(
            capacity=capacity,
            calculator=calculator,
            ic_lower_bound=ic_lower_bound,
            l1_alpha=l1_alpha,
            device=device
        )
        self.shadow_pool = ShadowAlphaPool(shadow_capacity)
        self.shadow_export_top_k = shadow_export_top_k
        self.shadow_rank_exact_top_n = (
            shadow_capacity if shadow_rank_exact_top_n is None else shadow_rank_exact_top_n
        )
        self.shadow_refit = shadow_refit

    def _on_alpha_evicted(
        self,
        expr: Expression,
        weight: float,
        single_ic: float,
        reason: str
    ) -> None:
        self.shadow_pool.add(
            expr=expr,
            single_ic=single_ic,
            evicted_weight=weight,
            evicted_at_eval=self.eval_cnt,
            reason=reason
        )

    def rank_shadow(self, exact_top_n: Optional[int] = None) -> List[ShadowAlphaRecord]:
        active_keys = {str(expr) for expr in self.exprs[:self.size] if expr is not None}
        records = [
            record for record in self.shadow_pool.records
            if record.expr_str not in active_keys
        ]
        if len(records) == 0:
            return []

        active_exprs: List[Expression] = [
            expr for expr in self.exprs[:self.size] if expr is not None
        ]
        for record in records:
            self._estimate_shadow_record(record, active_exprs)

        records.sort(key=self._shadow_sort_key, reverse=True)
        if exact_top_n is None:
            exact_top_n = self.shadow_rank_exact_top_n
        exact_top_n = max(0, min(exact_top_n, len(records)))

        base_ic = self.evaluate_ensemble()
        for record in records[:exact_top_n]:
            self._evaluate_shadow_delta(record, active_exprs, base_ic)

        self.shadow_pool.trim()
        records.sort(key=self._shadow_sort_key, reverse=True)
        return records

    def _estimate_shadow_record(
        self,
        record: ShadowAlphaRecord,
        active_exprs: List[Expression]
    ) -> None:
        record.delta_ic = None
        record.exact_evaluated = False
        record.last_error = None
        record.active_mutual_ics = None

        if len(active_exprs) == 0:
            record.approx_score = abs(record.single_ic)
            record.suggested_weight = record.single_ic
            return

        try:
            mutual_ics = self._calc_shadow_mutual_ics(record, active_exprs)
            residual_ic = record.single_ic - float(np.dot(self.weights, mutual_ics))
            record.approx_score = abs(residual_ic)
            record.suggested_weight = residual_ic
            record.active_mutual_ics = [float(value) for value in mutual_ics]
        except (OutOfDataRangeError, TypeError, ValueError) as exc:
            record.approx_score = float("-inf")
            record.suggested_weight = 0.
            record.last_error = f"{type(exc).__name__}: {exc}"

    def _evaluate_shadow_delta(
        self,
        record: ShadowAlphaRecord,
        active_exprs: List[Expression],
        base_ic: float
    ) -> None:
        try:
            mutual_ics = (
                np.array(record.active_mutual_ics)
                if record.active_mutual_ics is not None and len(record.active_mutual_ics) == self.size
                else self._calc_shadow_mutual_ics(record, active_exprs)
            )
            initial_weight = (
                record.suggested_weight
                if np.isfinite(record.suggested_weight)
                else record.evicted_weight
            )
            if self.shadow_refit:
                n = self.size
                single_ics = np.concatenate([
                    self.single_ics[:n],
                    np.array([record.single_ic])
                ])
                mutual_matrix = np.identity(n + 1)
                if n > 0:
                    mutual_matrix[:n, :n] = self._mutual_ics[:n, :n]
                    mutual_matrix[:n, n] = mutual_ics
                    mutual_matrix[n, :n] = mutual_ics
                initial_weights = np.concatenate([
                    self.weights,
                    np.array([initial_weight])
                ])
                weights = self._optimize_weights_from_stats(
                    single_ics,
                    mutual_matrix,
                    initial_weights
                )
            else:
                weights = np.concatenate([
                    self.weights,
                    np.array([initial_weight])
                ])
            new_ic = self.calculator.calc_pool_IC_ret(
                active_exprs + [record.expr],
                weights
            )
            record.delta_ic = float(new_ic - base_ic)
            record.suggested_weight = float(weights[-1])
            record.exact_evaluated = True
            record.last_error = None
        except (OutOfDataRangeError, TypeError, ValueError, RuntimeError) as exc:
            record.delta_ic = None
            record.exact_evaluated = False
            record.last_error = f"{type(exc).__name__}: {exc}"

    def _calc_shadow_mutual_ics(
        self,
        record: ShadowAlphaRecord,
        active_exprs: List[Expression]
    ) -> np.ndarray:
        return np.array([
            self.calculator.calc_mutual_IC(record.expr, expr)
            for expr in active_exprs
        ])

    def _shadow_sort_key(self, record: ShadowAlphaRecord) -> Any:
        if record.delta_ic is not None:
            score = record.delta_ic
        else:
            score = record.approx_score
        return score, record.approx_score, record.single_ic, record.evicted_at_eval

    def _active_export_indices(self) -> List[int]:
        return sorted(
            range(self.size),
            key=lambda i: abs(self.weights[i]),
            reverse=True
        )

    def to_json_dict(self) -> Dict[str, Any]:
        raw = super().to_json_dict()
        active_indices = self._active_export_indices()
        active_items = [
            {
                "rank": rank,
                "tier": "active",
                "expr": str(self.exprs[index]),
                "weight": float(self.weights[index]),
                "single_ic": float(self.single_ics[index]),
                "abs_weight": float(abs(self.weights[index])),
            }
            for rank, index in enumerate(active_indices, start=1)
        ]

        ranked_shadow = self.rank_shadow()
        shadow_start_rank = len(active_items) + 1
        shadow_items = [
            record.to_json_dict(rank)
            for rank, record in enumerate(ranked_shadow, start=shadow_start_rank)
        ]

        if self.shadow_export_top_k is None:
            shadow_export_count = len(shadow_items)
        else:
            shadow_export_count = max(0, self.shadow_export_top_k - len(active_items))
        exported_shadow = shadow_items[:shadow_export_count]

        raw.update({
            "active": active_items,
            "shadow": shadow_items,
            "ranked_exprs": (
                [item["expr"] for item in active_items] +
                [item["expr"] for item in exported_shadow]
            ),
            "ranked_weights": (
                [item["weight"] for item in active_items] +
                [item["suggested_weight"] for item in exported_shadow]
            ),
            "shadow_config": {
                "active_capacity": self.capacity,
                "shadow_capacity": self.shadow_pool.capacity,
                "shadow_export_top_k": self.shadow_export_top_k,
                "shadow_rank_exact_top_n": self.shadow_rank_exact_top_n,
                "shadow_refit": self.shadow_refit,
            }
        })
        return raw
