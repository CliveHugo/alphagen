from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .linear_alpha_pool import MseAlphaPool
from ..data.calculator import AlphaCalculator, TensorAlphaCalculator
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
        shadow_refit: bool = False,
        shadow_rank_batch_size: int = 16
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
        self.shadow_rank_batch_size = shadow_rank_batch_size

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

    def _calc_ics(
        self,
        expr: Expression,
        ic_mut_threshold: Optional[float] = None
    ) -> Tuple[float, Optional[List[float]]]:
        if not isinstance(self.calculator, TensorAlphaCalculator):
            return super()._calc_ics(expr, ic_mut_threshold)

        calculator = self.calculator
        try:
            value = calculator.evaluate_alpha(expr)
            target = calculator.target.to(device=value.device, dtype=value.dtype)
            if self._contains_nan(value, target):
                return super()._calc_ics(expr, ic_mut_threshold)

            single_ic = float(self._mean_daily_pearson_to_target_no_nan(value[None, :, :], target)[0].item())
            if not self._under_thres_alpha and single_ic < self._ic_lower_bound:
                return single_ic, None

            if self.size == 0:
                return single_ic, []

            active_values = self._evaluate_active_values(self._active_exprs(), calculator, value[None, :, :])
            if self._contains_nan(active_values):
                return super()._calc_ics(expr, ic_mut_threshold)

            mutual_ics = self._mean_daily_pearson_pairwise_no_nan(
                value[None, :, :],
                active_values
            )[0]
            if ic_mut_threshold is not None and (mutual_ics > ic_mut_threshold).any().item():
                return single_ic, None
            return single_ic, [float(value) for value in mutual_ics.detach().cpu().numpy()]
        except (TypeError, ValueError):
            return super()._calc_ics(expr, ic_mut_threshold)

    def rank_shadow(self, exact_top_n: Optional[int] = None) -> List[ShadowAlphaRecord]:
        active_keys = {str(expr) for expr in self.exprs[:self.size] if expr is not None}
        records = [
            record for record in self.shadow_pool.records
            if record.expr_str not in active_keys
        ]
        if len(records) == 0:
            return []

        if isinstance(self.calculator, TensorAlphaCalculator):
            return self._rank_shadow_tensor(records, exact_top_n)

        return self._rank_shadow_generic(records, exact_top_n)

    def _rank_shadow_generic(
        self,
        records: List[ShadowAlphaRecord],
        exact_top_n: Optional[int]
    ) -> List[ShadowAlphaRecord]:
        active_exprs = self._active_exprs()

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

    def _rank_shadow_tensor(
        self,
        records: List[ShadowAlphaRecord],
        exact_top_n: Optional[int]
    ) -> List[ShadowAlphaRecord]:
        calculator = self.calculator
        assert isinstance(calculator, TensorAlphaCalculator)

        active_exprs = self._active_exprs()
        target = calculator.target
        active_values = self._evaluate_active_values(active_exprs, calculator, target[None, :, :])
        target = target.to(device=active_values.device, dtype=active_values.dtype)
        if self._contains_nan(active_values, target):
            return self._rank_shadow_generic(records, exact_top_n)

        with torch.no_grad():
            if len(active_exprs) == 0:
                base_value = torch.zeros_like(target)
                base_ic = 0.
            else:
                active_weights = torch.tensor(
                    self.weights,
                    device=active_values.device,
                    dtype=active_values.dtype
                )
                base_value = (active_weights[:, None, None] * active_values).sum(dim=0)
                base_ic = self._mean_daily_pearson_to_target_no_nan(
                    base_value[None, :, :],
                    target
                )[0].item()

            valid_records = []
            for record_chunk, shadow_values in self._iter_shadow_value_chunks(records, calculator, target):
                if self._contains_nan(shadow_values):
                    return self._rank_shadow_generic(records, exact_top_n)
                valid_records.extend(record_chunk)
                if len(active_exprs) == 0:
                    mutual_ics = torch.empty(
                        (len(record_chunk), 0),
                        device=shadow_values.device,
                        dtype=shadow_values.dtype
                    )
                    residual_ics = torch.tensor(
                        [record.single_ic for record in record_chunk],
                        device=shadow_values.device,
                        dtype=shadow_values.dtype
                    )
                else:
                    mutual_ics = self._mean_daily_pearson_pairwise_no_nan(
                        shadow_values,
                        active_values
                    )
                    single_ics = torch.tensor(
                        [record.single_ic for record in record_chunk],
                        device=shadow_values.device,
                        dtype=shadow_values.dtype
                    )
                    residual_ics = single_ics - mutual_ics.matmul(active_weights)

                mutual_np = mutual_ics.detach().cpu().numpy()
                residual_np = residual_ics.detach().cpu().numpy()
                for i, record in enumerate(record_chunk):
                    residual_ic = float(residual_np[i])
                    record.delta_ic = None
                    record.exact_evaluated = False
                    record.last_error = None
                    record.approx_score = abs(residual_ic)
                    record.suggested_weight = residual_ic
                    record.active_mutual_ics = [float(value) for value in mutual_np[i]]

            if len(valid_records) == 0:
                return []

            valid_records.sort(key=self._shadow_sort_key, reverse=True)
            if exact_top_n is None:
                exact_top_n = self.shadow_rank_exact_top_n
            exact_top_n = max(0, min(exact_top_n, len(valid_records)))

            if self.shadow_refit:
                for record in valid_records[:exact_top_n]:
                    self._evaluate_shadow_delta(record, active_exprs, base_ic)
            elif exact_top_n > 0:
                exact_records = valid_records[:exact_top_n]
                for record_chunk, shadow_values in self._iter_shadow_value_chunks(exact_records, calculator, target):
                    selected_weights = torch.tensor(
                        [record.suggested_weight for record in record_chunk],
                        device=shadow_values.device,
                        dtype=shadow_values.dtype
                    )
                    candidate_values = base_value[None, :, :] + selected_weights[:, None, None] * shadow_values
                    new_ics = self._mean_daily_pearson_to_target_no_nan(candidate_values, target)
                    deltas = (new_ics - base_ic).detach().cpu().numpy()
                    for record, delta_ic in zip(record_chunk, deltas):
                        record.delta_ic = float(delta_ic)
                        record.exact_evaluated = True
                        record.last_error = None

        self.shadow_pool.trim()
        valid_records.sort(key=self._shadow_sort_key, reverse=True)
        return valid_records

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

    def _active_exprs(self) -> List[Expression]:
        return [
            expr for expr in self.exprs[:self.size] if expr is not None
        ]

    def _evaluate_shadow_values(
        self,
        records: List[ShadowAlphaRecord],
        calculator: TensorAlphaCalculator
    ) -> Tuple[List[ShadowAlphaRecord], torch.Tensor]:
        valid_records = []
        values = []
        for record in records:
            try:
                values.append(calculator.evaluate_alpha(record.expr))
                valid_records.append(record)
            except (OutOfDataRangeError, TypeError, ValueError, RuntimeError) as exc:
                record.approx_score = float("-inf")
                record.suggested_weight = 0.
                record.delta_ic = None
                record.exact_evaluated = False
                record.last_error = f"{type(exc).__name__}: {exc}"
                record.active_mutual_ics = None

        if len(values) == 0:
            return [], torch.empty(0, device=self.device)
        return valid_records, torch.stack(values)

    def _evaluate_active_values(
        self,
        active_exprs: List[Expression],
        calculator: TensorAlphaCalculator,
        shadow_values: torch.Tensor
    ) -> torch.Tensor:
        if len(active_exprs) == 0:
            return torch.empty(
                (0, *shadow_values.shape[1:]),
                device=shadow_values.device,
                dtype=shadow_values.dtype
            )
        return torch.stack([
            calculator.evaluate_alpha(expr).to(device=shadow_values.device, dtype=shadow_values.dtype)
            for expr in active_exprs
        ])

    def _iter_shadow_value_chunks(
        self,
        records: List[ShadowAlphaRecord],
        calculator: TensorAlphaCalculator,
        reference: torch.Tensor
    ):
        batch_size = max(1, self.shadow_rank_batch_size)
        for start in range(0, len(records), batch_size):
            record_chunk = records[start:start + batch_size]
            valid_records, values = self._evaluate_shadow_values(record_chunk, calculator)
            if len(valid_records) == 0:
                continue
            yield (
                valid_records,
                values.to(device=reference.device, dtype=reference.dtype)
            )

    def _contains_nan(self, *values: torch.Tensor) -> bool:
        return any(value.numel() > 0 and torch.isnan(value).any().item() for value in values)

    def _mean_daily_pearson_pairwise_no_nan(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        chunk_size: int = 16
    ) -> torch.Tensor:
        if ys.shape[0] == 0:
            return torch.empty((xs.shape[0], 0), device=xs.device, dtype=xs.dtype)

        n_stocks = xs.shape[-1]
        y_mean = ys.mean(dim=2)
        y_std = ((ys - y_mean[:, :, None]) ** 2).mean(dim=2).sqrt()
        outputs = []
        for start in range(0, xs.shape[0], chunk_size):
            chunk = xs[start:start + chunk_size]
            x_mean = chunk.mean(dim=2)
            x_std = ((chunk - x_mean[:, :, None]) ** 2).mean(dim=2).sqrt()
            prod_mean = torch.einsum("sdn,adn->sad", chunk, ys) / n_stocks
            cov = prod_mean - x_mean[:, None, :] * y_mean[None, :, :]
            stdmul = x_std[:, None, :] * y_std[None, :, :]
            stdmul = torch.where(
                (x_std[:, None, :] < 1e-3) | (y_std[None, :, :] < 1e-3),
                torch.ones_like(stdmul),
                stdmul
            )
            outputs.append((cov / stdmul).mean(dim=2))
        return torch.cat(outputs, dim=0)

    def _mean_daily_pearson_to_target_no_nan(
        self,
        xs: torch.Tensor,
        target: torch.Tensor,
        chunk_size: int = 32
    ) -> torch.Tensor:
        y_mean = target.mean(dim=1)
        y_std = ((target - y_mean[:, None]) ** 2).mean(dim=1).sqrt()
        outputs = []
        for start in range(0, xs.shape[0], chunk_size):
            chunk = xs[start:start + chunk_size]
            x_mean = chunk.mean(dim=2)
            x_std = ((chunk - x_mean[:, :, None]) ** 2).mean(dim=2).sqrt()
            prod_mean = (chunk * target[None, :, :]).mean(dim=2)
            cov = prod_mean - x_mean * y_mean[None, :]
            stdmul = x_std * y_std[None, :]
            stdmul = torch.where(
                (x_std < 1e-3) | (y_std[None, :] < 1e-3),
                torch.ones_like(stdmul),
                stdmul
            )
            outputs.append((cov / stdmul).mean(dim=1))
        return torch.cat(outputs, dim=0)

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
                "shadow_rank_batch_size": self.shadow_rank_batch_size,
            }
        })
        return raw
