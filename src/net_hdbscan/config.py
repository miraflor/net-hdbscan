"""Parameter objects and per-group overrides."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .sparse_hdbscan import validate_parameters
from .network import DEFAULT_MAX_NEIGHBOR_PAIRS, DEFAULT_VERTEX_DIGITS


def _positive_finite(name: str, value) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return value


@dataclass(frozen=True)
class HDBSCANConfig:
    """Clustering parameters. Distances are in network-CRS units (usually metres).

    max_distance
        Largest shortest-path distance that is searched. Pairs farther apart
        are treated as unconnected, like pairs on disconnected parts of the network. No
        cluster can be held together by links longer than this.
    min_cluster_size
        Smallest number of observations that counts as a cluster.
    min_samples
        Number of observations (the observation itself included) that must
        lie within an observation's core distance. Larger values make the
        density estimate smoother and label more observations as noise.
        Default: ``min_cluster_size``.
    cluster_selection_method
        ``"eom"`` (Excess of Mass, default) or ``"leaf"``.
    cluster_selection_epsilon
        Clusters that separate only at distances below this value are kept
        together (HDBSCAN(epsilon-hat)).
    max_cluster_size
        With ``"eom"``, clusters larger than this are not selected.
    allow_single_cluster
        Allow the result to be one cluster that contains all others.
    core_distance_floor
        Core distances below this value are raised to it. 0 (default) gives
        standard HDBSCAN. Use a positive value to limit the effect of many
        observations stacked on one position.
    duplicates
        ``"count"`` (default): observations at one snapped position count
        individually (exactly as HDBSCAN on the observations).
        ``"once"``: each distinct position counts once; all observations at
        the position receive its label.
    noise_policy
        ``"exclude"``: noise observations get no cluster ID.
        ``"singleton"``: each noise position gets its own cluster ID.
    max_snap_distance
        Raise an error if any observation lies farther than this from the
        nearest network arc. ``None`` disables the check.
    max_neighbor_pairs
        Safety limit on stored pairs of positions within the active search radius.
    distance_mode
        ``"fixed"`` (default) uses ``max_distance`` exactly. ``"adaptive"``
        treats it as a hard ceiling and grows a smaller search radius until
        the flat result reaches a stable low-truncation plateau, the ceiling
        is reached, or the pair budget prevents a larger trial.
    min_distance
        Optional first radius in adaptive mode. If omitted, a geometric
        ladder is derived from ``max_distance``, ``distance_growth`` and
        ``distance_steps``.
    distance_growth
        Geometric growth factor for adaptive radii.
    distance_steps
        Nominal number of adaptive radii when ``min_distance`` is omitted.
    core_truncation_tolerance
        Maximum share of otherwise-resolvable observations whose core
        distance may remain beyond the adaptive radius when declaring a
        stable result.
    stability_tolerance
        Absolute tolerance for membership values in the adaptive stability
        comparison.
    """

    max_distance: float
    min_cluster_size: int = 5
    min_samples: int | None = None
    cluster_selection_method: str = "eom"
    cluster_selection_epsilon: float = 0.0
    max_cluster_size: int | None = None
    allow_single_cluster: bool = False
    core_distance_floor: float = 0.0
    duplicates: str = "count"
    noise_policy: str = "exclude"
    max_snap_distance: float | None = None
    max_neighbor_pairs: int = DEFAULT_MAX_NEIGHBOR_PAIRS
    distance_mode: str = "fixed"
    min_distance: float | None = None
    distance_growth: float = 1.5
    distance_steps: int = 4
    core_truncation_tolerance: float = 0.01
    stability_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_distance", _positive_finite("max_distance", self.max_distance))
        validate_parameters(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            max_cluster_size=self.max_cluster_size,
        )
        floor = float(self.core_distance_floor)
        if not (math.isfinite(floor) and 0 <= floor < self.max_distance):
            raise ValueError("core_distance_floor must be >= 0 and smaller than max_distance")
        if not isinstance(self.allow_single_cluster, (bool, np.bool_)):
            raise ValueError("allow_single_cluster must be True or False")
        if self.duplicates not in {"count", "once"}:
            raise ValueError("duplicates must be 'count' or 'once'")
        if self.noise_policy not in {"exclude", "singleton"}:
            raise ValueError("noise_policy must be 'exclude' or 'singleton'")
        if self.max_snap_distance is not None:
            limit = float(self.max_snap_distance)
            if math.isnan(limit) or limit < 0:
                raise ValueError("max_snap_distance must be None or a number >= 0")
        if (
            isinstance(self.max_neighbor_pairs, bool)
            or not isinstance(self.max_neighbor_pairs, (int, np.integer))
            or self.max_neighbor_pairs < 1
        ):
            raise ValueError("max_neighbor_pairs must be an integer >= 1")
        if self.distance_mode not in {"fixed", "adaptive"}:
            raise ValueError("distance_mode must be 'fixed' or 'adaptive'")
        if self.min_distance is not None:
            minimum = _positive_finite("min_distance", self.min_distance)
            if minimum > self.max_distance:
                raise ValueError("min_distance must be <= max_distance")
            object.__setattr__(self, "min_distance", minimum)
        growth = float(self.distance_growth)
        if not (math.isfinite(growth) and growth > 1):
            raise ValueError("distance_growth must be a finite number > 1")
        object.__setattr__(self, "distance_growth", growth)
        if (
            isinstance(self.distance_steps, bool)
            or not isinstance(self.distance_steps, (int, np.integer))
            or self.distance_steps < 2
        ):
            raise ValueError("distance_steps must be an integer >= 2")
        tol = float(self.core_truncation_tolerance)
        if not (math.isfinite(tol) and 0 <= tol <= 1):
            raise ValueError("core_truncation_tolerance must be between 0 and 1")
        object.__setattr__(self, "core_truncation_tolerance", tol)
        stability = float(self.stability_tolerance)
        if not (math.isfinite(stability) and stability >= 0):
            raise ValueError("stability_tolerance must be a finite number >= 0")
        object.__setattr__(self, "stability_tolerance", stability)

    @property
    def effective_min_samples(self) -> int:
        return int(self.min_cluster_size if self.min_samples is None else self.min_samples)

    def replace(self, **changes: Any) -> HDBSCANConfig:
        return dataclasses.replace(self, **changes)

    def as_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["effective_min_samples"] = self.effective_min_samples
        return out


# ---------------------------------------------------------------------------
# Per-group overrides
# ---------------------------------------------------------------------------

# Internal keys are deliberately impossible to confuse with ordinary group
# strings. User-facing missing-group tokens stay ``__null__`` and ``__blank__``.
NULL_GROUP_TOKEN = "__null__"
BLANK_GROUP_TOKEN = "__blank__"
NULL_GROUP_KEY = "\0net_hdbscan:null"
BLANK_GROUP_KEY = "\0net_hdbscan:blank"

_INT_FIELDS = {"min_cluster_size", "min_samples", "max_cluster_size", "distance_steps"}
_FLOAT_FIELDS = {
    "max_distance",
    "min_distance",
    "distance_growth",
    "core_truncation_tolerance",
    "stability_tolerance",
    "cluster_selection_epsilon",
    "core_distance_floor",
}
_BOOL_FIELDS = {"allow_single_cluster"}
_TEXT_FIELDS = {"cluster_selection_method", "duplicates", "distance_mode"}
GROUP_OVERRIDE_FIELDS = tuple(sorted(_INT_FIELDS | _FLOAT_FIELDS | _BOOL_FIELDS | _TEXT_FIELDS))


def group_key(value) -> str:
    """Internal key of a group value, keeping missing values distinct from text."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return NULL_GROUP_KEY
    text = str(value)
    return BLANK_GROUP_KEY if text.strip() == "" else text


def group_display(key: str) -> str:
    """Unambiguous user-facing text for an internal group key."""
    if key == NULL_GROUP_KEY:
        return NULL_GROUP_TOKEN
    if key == BLANK_GROUP_KEY:
        return BLANK_GROUP_TOKEN
    # These two literal strings would otherwise look exactly like the missing
    # groups in summaries and filenames. Prefix only the colliding literals.
    if key in {NULL_GROUP_TOKEN, BLANK_GROUP_TOKEN}:
        return f"literal:{key}"
    return str(key)


def _group_param_key(text: str) -> str:
    """Parse one group-parameter CSV key. ``literal:`` escapes reserved tokens."""
    if text == NULL_GROUP_TOKEN:
        return NULL_GROUP_KEY
    if text == BLANK_GROUP_TOKEN:
        return BLANK_GROUP_KEY
    if text.startswith("literal:"):
        return text[len("literal:") :]
    return text


def _convert(field: str, text: str):
    if field in _INT_FIELDS:
        number = float(text)
        if not number.is_integer():
            raise ValueError(f"{field} must be an integer, got {text!r}")
        return int(number)
    if field in _FLOAT_FIELDS:
        return float(text)
    if field in _BOOL_FIELDS:
        lowered = text.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        raise ValueError(f"{field} must be true or false, got {text!r}")
    return text.strip()


def read_group_params(path: str | Path, group_col: str) -> dict[str, dict[str, Any]]:
    """Read per-group parameter overrides from a CSV file.

    The first column must be named ``group_col`` (or ``group``) and holds
    group values as text; use ``__null__`` for missing values and
    ``__blank__`` for empty text. A literal group with either of those exact
    names is written as ``literal:__null__`` or ``literal:__blank__``. Other columns must be names from
    ``GROUP_OVERRIDE_FIELDS``. An empty cell keeps the run-wide value.
    """
    # utf-8-sig also accepts the byte-order mark that Excel writes on Windows.
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if len(table.columns) == 0:
        raise ValueError("group parameter file has no columns")
    key_col = table.columns[0]
    if key_col not in {group_col, "group"}:
        raise ValueError(f"first column of the group parameter file must be {group_col!r} or 'group', got {key_col!r}")
    unknown = sorted(set(table.columns[1:]) - set(GROUP_OVERRIDE_FIELDS))
    if unknown:
        raise ValueError(f"unknown columns in group parameter file: {unknown}; allowed: {list(GROUP_OVERRIDE_FIELDS)}")
    overrides: dict[str, dict[str, Any]] = {}
    semantic_keys: set[str] = set()
    for _, row in table.iterrows():
        raw_key = row[key_col]
        if raw_key == "":
            raise ValueError("empty group value in parameter file; write __blank__ for empty text")
        semantic = _group_param_key(raw_key)
        if semantic in semantic_keys:
            raise ValueError(f"group parameter file lists group {group_display(semantic)!r} more than once")
        semantic_keys.add(semantic)
        values = {}
        for field in table.columns[1:]:
            text = row[field]
            if text.strip() != "":
                values[field] = _convert(field, text)
        # Preserve the readable CSV spelling in the returned public mapping.
        # cluster_files_by_column normalizes these keys before matching groups.
        overrides[raw_key] = values
    return overrides


def normalize_group_params(overrides: Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Normalize public group-override keys to the pipeline's internal keys.

    ``__null__`` and ``__blank__`` target missing values. Prefix ``literal:``
    to target an ordinary string with a reserved name. Internal sentinel keys
    are also accepted for advanced callers.
    """
    if not overrides:
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for raw_key, values in overrides.items():
        key = raw_key if raw_key in {NULL_GROUP_KEY, BLANK_GROUP_KEY} else _group_param_key(str(raw_key))
        if key in normalized:
            raise ValueError(f"group parameter overrides list group {group_display(key)!r} more than once")
        normalized[key] = dict(values)
    return normalized



def read_group_universe(path: str | Path, group_col: str) -> list[str]:
    """Read the declared universe of group values from a one-column CSV.

    The column must be named ``group_col`` or ``group``. Values are kept as
    text. Missing/blank universe entries are rejected: use
    ``missing_group_policy='include'`` for actual missing observations.
    """
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if len(table.columns) != 1:
        raise ValueError("group universe file must contain exactly one column")
    key_col = table.columns[0]
    if key_col not in {group_col, "group"}:
        raise ValueError(f"group universe column must be {group_col!r} or 'group', got {key_col!r}")
    values = [group_key(v) for v in table[key_col].tolist()]
    if any(v in {NULL_GROUP_KEY, BLANK_GROUP_KEY} for v in values):
        raise ValueError("group universe cannot contain null/blank values")
    if len(values) != len(set(values)):
        raise ValueError("group universe lists a group more than once")
    return values


def resolve_group_config(base: HDBSCANConfig, overrides: Mapping[str, Mapping[str, Any]] | None, key: str) -> HDBSCANConfig:
    if not overrides or key not in overrides:
        return base
    changes = dict(overrides[key])
    unknown = sorted(set(changes) - set(GROUP_OVERRIDE_FIELDS))
    if unknown:
        raise ValueError(f"group {group_display(key)!r}: unknown override fields {unknown}")
    try:
        return base.replace(**changes)
    except ValueError as exc:
        raise ValueError(f"group {group_display(key)!r}: {exc}") from exc


__all__ = [
    "DEFAULT_VERTEX_DIGITS",
    "GROUP_OVERRIDE_FIELDS",
    "HDBSCANConfig",
    "group_display",
    "group_key",
    "normalize_group_params",
    "read_group_params",
    "read_group_universe",
    "resolve_group_config",
]
