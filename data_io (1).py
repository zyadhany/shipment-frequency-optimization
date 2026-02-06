"""Load + validate CSV inputs for the shipment planner.

Schema rule per file: list of (col, default)
- default is None  -> required column; missing cells drop the row
- default provided -> optional column; if missing create with default; missing cells filled
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd


# (col, default). default=None => required.
ORDERS_SCHEMA = [
    ("order_id", None),
    ("order_date", None),
    ("branch_id", None),
    ("sku_id", None),
    ("demand_qty", None),
    # Default rule: if missing, due_date = order_date + 10 days
    ("due_date", pd.NA),
    ("priority", ""),  # optional
]

INVENTORY_SCHEMA = [
    ("location_id", None),
    ("sku_id", None),
    ("on_hand_qty", None),
    ("safety_stock", 0),  # optional
]

CAPACITY_SCHEMA = [
    ("location_id", None),
    ("date", None),
    ("max_units", None),
    ("max_shipments", None),
    # max_weight_kg is optional; validated only if present in the file
]

TRANSPORT_SCHEMA = [
    ("from_location_id", None),
    ("to_location_id", None),
    ("mode", None),
    ("lead_time_days", None),
    ("cost_per_unit", None),
    ("fixed_cost", None),
]


def load_orders(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_inventory(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_capacity(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_transport_costs(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _naize(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(r"^\s*$", pd.NA, regex=True) if not df.empty else df.copy()


def _apply_schema(df: pd.DataFrame, schema: List[Tuple[str, object]], name: str) -> tuple[pd.DataFrame, int]:
    df = _naize(df)
    req = [c for c, d in schema if d is None]
    opt = {c: d for c, d in schema if d is not None}

    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: missing required columns: {miss}")

    for c, d in opt.items():
        if c not in df.columns:
            df[c] = d
        else:
            df[c] = df[c].fillna(d)

    before = len(df)
    df = df.dropna(subset=req) if req else df
    return df, before - len(df)


def _parse_dates(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_datetime(out[c], errors="coerce").dt.normalize()
        if out[c].isna().any():
            bad = out.index[out[c].isna()].tolist()[:10]
            raise ValueError(f"{name}: invalid date in '{c}' (bad rows {bad}, up to 10)")
    return out


def _num_nonneg(df: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if out[c].isna().any():
            bad = out.index[out[c].isna()].tolist()[:10]
            raise ValueError(f"{name}: '{c}' must be numeric and non-null (bad rows {bad}, up to 10)")
        if (out[c] < 0).any():
            bad = out.index[out[c] < 0].tolist()[:10]
            raise ValueError(f"{name}: '{c}' must be >= 0 (bad rows {bad}, up to 10)")
    return out


def _opt_num_nonneg(df: pd.DataFrame, col: str, name: str, default: float = 0.0) -> pd.DataFrame:
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    if (out[col] < 0).any():
        bad = out.index[out[col] < 0].tolist()[:10]
        raise ValueError(f"{name}: '{col}' must be >= 0 (bad rows {bad}, up to 10)")
    return out


def _int_nonneg(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[col].isna().any():
        bad = out.index[out[col].isna()].tolist()[:10]
        raise ValueError(f"{name}: '{col}' must be integer >= 0 (bad rows {bad}, up to 10)")
    if (out[col] < 0).any() or (out[col] % 1 != 0).any():
        bad = out.index[(out[col] < 0) | (out[col] % 1 != 0)].tolist()[:10]
        raise ValueError(f"{name}: '{col}' must be integer >= 0 (bad rows {bad}, up to 10)")
    out[col] = out[col].astype(int)
    return out


def _detect_weight_col(inv: pd.DataFrame) -> Optional[str]:
    for c in ("weight_kg", "unit_weight_kg", "unit_weight", "weight"):
        if c in inv.columns:
            return c
    return None


@dataclass(frozen=True)
class ValidatedInputs:
    orders: pd.DataFrame
    inventory: pd.DataFrame
    capacity: pd.DataFrame
    transport_costs: pd.DataFrame
    weight_col: Optional[str]
    warnings: List[str]


def validate_all(
    orders: pd.DataFrame,
    inventory: pd.DataFrame,
    capacity: pd.DataFrame,
    transport_costs: pd.DataFrame,
) -> ValidatedInputs:
    warnings: List[str] = []

    orders, do = _apply_schema(orders, ORDERS_SCHEMA, "orders")
    inventory, di = _apply_schema(inventory, INVENTORY_SCHEMA, "inventory")
    capacity, dc = _apply_schema(capacity, CAPACITY_SCHEMA, "capacity")
    transport_costs, dt = _apply_schema(transport_costs, TRANSPORT_SCHEMA, "transport_costs")

    for name, dropped in (("orders", do), ("inventory", di), ("capacity", dc), ("transport_costs", dt)):
        if dropped:
            warnings.append(f"{name}: dropped {dropped} row(s) with missing required cell(s)")

    # order_date is required; due_date may be missing and will be defaulted.
    orders = _parse_dates(orders, ["order_date"], "orders")
    # If due_date is provided but invalid, that's an error. Only truly-missing due_date is defaulted.
    _raw_due = orders["due_date"]
    _due = pd.to_datetime(_raw_due, errors="coerce").dt.normalize()
    bad_due_parse = _raw_due.notna() & _due.isna()
    if bad_due_parse.any():
        bad = orders.index[bad_due_parse].tolist()[:10]
        raise ValueError(f"orders: invalid date in 'due_date' for rows {bad} (up to 10)")
    orders["due_date"] = _due
    miss_due = orders["due_date"].isna()
    if miss_due.any():
        orders.loc[miss_due, "due_date"] = orders.loc[miss_due, "order_date"] + pd.Timedelta(days=10)
        warnings.append(f"orders: filled {int(miss_due.sum())} missing due_date with order_date+10d")
    if orders["due_date"].isna().any():
        bad = orders.index[orders["due_date"].isna()].tolist()[:10]
        raise ValueError(f"orders: due_date is required after defaulting; still missing for rows {bad} (up to 10)")
    capacity = _parse_dates(capacity, ["date"], "capacity")

    orders = _num_nonneg(orders, ["demand_qty"], "orders")
    inventory = _num_nonneg(inventory, ["on_hand_qty", "safety_stock"], "inventory")
    capacity = _num_nonneg(capacity, ["max_units", "max_shipments"], "capacity")
    capacity = _opt_num_nonneg(capacity, "max_weight_kg", "capacity", default=0.0)

    transport_costs = _int_nonneg(transport_costs, "lead_time_days", "transport_costs")
    transport_costs = _num_nonneg(transport_costs, ["cost_per_unit", "fixed_cost"], "transport_costs")

    # Normalize id-like columns to strings (so joins are stable)
    for df, cols in (
        (orders, ["order_id", "branch_id", "sku_id", "priority"]),
        (inventory, ["location_id", "sku_id"]),
        (capacity, ["location_id"]),
        (transport_costs, ["from_location_id", "to_location_id", "mode"]),
    ):
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str)

    # Hard checks: branch reachability + SKU coverage
    branches = set(orders["branch_id"])
    to_locs = set(transport_costs["to_location_id"])
    miss_br = sorted(branches - to_locs)
    if miss_br:
        raise ValueError("orders: branch_id(s) unreachable (not in transport_costs.to_location_id): " + ", ".join(miss_br[:20]))

    miss_sku = sorted(set(orders["sku_id"]) - set(inventory["sku_id"]))
    if miss_sku:
        raise ValueError("orders: sku_id(s) not present in inventory.sku_id: " + ", ".join(miss_sku[:20]))

    # Warnings
    bad_due = orders[orders["due_date"] < orders["order_date"]]
    if not bad_due.empty:
        warnings.append(f"{len(bad_due)} order line(s) have due_date < order_date (example order_id(s): " + ", ".join(bad_due["order_id"].head(5).tolist()) + ")")

    cap_min, cap_max = capacity["date"].min(), capacity["date"].max()
    out_rng = orders[(orders["due_date"] < cap_min) | (orders["due_date"] > cap_max)]
    if not out_rng.empty:
        warnings.append(f"{len(out_rng)} order line(s) have due_date outside capacity date range [{cap_min.date()}..{cap_max.date()}].")

    inv_locs = set(inventory["location_id"])
    reachable_to = set(transport_costs.loc[transport_costs["from_location_id"].isin(inv_locs), "to_location_id"])
    no_lane = sorted(branches - reachable_to)
    if no_lane:
        warnings.append("No lane exists from any inventory location to some branch_id(s): " + ", ".join(no_lane[:20]))

    weight_col = _detect_weight_col(inventory)
    has_weight_cap = ("max_weight_kg" in capacity.columns) and (capacity["max_weight_kg"] > 0).any()
    if has_weight_cap and weight_col is None:
        warnings.append("Capacity has max_weight_kg but inventory has no weight column; weight constraints will be ignored.")
    if weight_col is not None and not has_weight_cap:
        warnings.append(f"Inventory has '{weight_col}' but capacity has no positive max_weight_kg; weight constraints will be ignored.")
    if weight_col is not None:
        inventory = _num_nonneg(inventory, [weight_col], "inventory")

    return ValidatedInputs(orders, inventory, capacity, transport_costs, weight_col, warnings)
