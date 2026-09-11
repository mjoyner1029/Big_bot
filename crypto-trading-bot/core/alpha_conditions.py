"""Machine-readable alpha condition DSL.

Alphas declare entry/exit conditions as constrained, validated expressions —
NEVER as natural language or executable Python. LLM-generated hypotheses must
be expressed in this DSL and are strictly validated before storage.

Condition shape (JSON-safe):
    {"feature": "relative_volume_20d", "op": ">",  "value": 3}
    {"feature": "day_of_week",         "op": "==", "value": "THURSDAY"}
    {"feature": "market_regime",       "op": "IN", "value": ["BULL", "HIGH_VOL"]}
    {"feature": "gap_pct",             "op": "BETWEEN", "value": [-0.08, -0.03]}

A condition set is a list of conditions ANDed together.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)

VALID_OPS = frozenset({">", "<", ">=", "<=", "==", "!=", "IN", "BETWEEN"})

_SCALAR_TYPES = (int, float, str, bool)


class ConditionValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"feature": self.feature, "op": self.op, "value": self.value}


def validate_condition(raw: Dict[str, Any]) -> Condition:
    """Strict validation: known feature, known operator, sane value types."""
    if not isinstance(raw, dict):
        raise ConditionValidationError(f"Condition must be a dict, got {type(raw)}")
    extra = set(raw) - {"feature", "op", "value"}
    if extra:
        raise ConditionValidationError(f"Unknown condition keys: {extra}")

    feature = raw.get("feature")
    op = raw.get("op")
    value = raw.get("value")

    if not isinstance(feature, str) or not FeatureRegistry.is_valid_feature(feature):
        raise ConditionValidationError(
            f"Unknown feature '{feature}' — must be one of the FeatureRegistry names")
    if op not in VALID_OPS:
        raise ConditionValidationError(f"Unknown operator '{op}' — allowed: {sorted(VALID_OPS)}")

    if op == "IN":
        if not isinstance(value, (list, tuple)) or not value or \
                not all(isinstance(v, _SCALAR_TYPES) for v in value):
            raise ConditionValidationError("IN requires a non-empty list of scalars")
    elif op == "BETWEEN":
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
                or value[0] > value[1]):
            raise ConditionValidationError("BETWEEN requires [low, high] numeric pair")
    else:
        if not isinstance(value, _SCALAR_TYPES):
            raise ConditionValidationError(f"{op} requires a scalar value")
        if op in {">", "<", ">=", "<="} and isinstance(value, (str, bool)):
            raise ConditionValidationError(f"{op} requires a numeric value")

    return Condition(feature=feature, op=op, value=value)


def validate_conditions(raw_list: Sequence[Dict[str, Any]]) -> List[Condition]:
    if not isinstance(raw_list, (list, tuple)):
        raise ConditionValidationError("Conditions must be a list")
    return [validate_condition(r) for r in raw_list]


def parse_conditions_json(payload: Optional[str]) -> List[Condition]:
    """Parse + validate conditions stored as JSON. Empty/None -> []."""
    if not payload:
        return []
    data = json.loads(payload) if isinstance(payload, str) else payload
    return validate_conditions(data)


def conditions_to_json(conditions: Sequence[Condition]) -> str:
    return json.dumps([c.to_dict() for c in conditions])


def evaluate_condition(cond: Condition, features: Dict[str, Any]) -> Tuple[bool, str]:
    """Evaluate one condition against computed features.

    Missing/None feature values FAIL the condition (abstain on unknown data).
    Returns (passed, reason).
    """
    actual = features.get(cond.feature)
    if actual is None:
        return False, f"{cond.feature} unavailable"

    try:
        if cond.op == "IN":
            ok = _norm(actual) in {_norm(v) for v in cond.value}
        elif cond.op == "BETWEEN":
            ok = float(cond.value[0]) <= float(actual) <= float(cond.value[1])
        elif cond.op == "==":
            ok = _norm(actual) == _norm(cond.value)
        elif cond.op == "!=":
            ok = _norm(actual) != _norm(cond.value)
        elif cond.op == ">":
            ok = float(actual) > float(cond.value)
        elif cond.op == "<":
            ok = float(actual) < float(cond.value)
        elif cond.op == ">=":
            ok = float(actual) >= float(cond.value)
        elif cond.op == "<=":
            ok = float(actual) <= float(cond.value)
        else:  # unreachable after validation
            return False, f"unsupported op {cond.op}"
    except (TypeError, ValueError) as e:
        return False, f"{cond.feature} type error: {e}"

    detail = f"{cond.feature}={actual!r} {cond.op} {cond.value!r}"
    return ok, detail


def evaluate_conditions(
    conditions: Sequence[Condition],
    features: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """AND-evaluate a condition set. Returns (all_passed, per-condition details)."""
    details: List[str] = []
    all_ok = True
    for cond in conditions:
        ok, detail = evaluate_condition(cond, features)
        details.append(f"{'PASS' if ok else 'FAIL'}: {detail}")
        if not ok:
            all_ok = False
    return all_ok, details


def _norm(v: Any) -> Any:
    return v.upper() if isinstance(v, str) else v
