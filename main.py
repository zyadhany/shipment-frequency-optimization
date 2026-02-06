"""Deterministic heuristic planner.

Reads:
  - /mnt/data/orders.csv
  - /mnt/data/inventory.csv
  - /mnt/data/capacity.csv
  - /mnt/data/transport_costs.csv

Writes:
  - /mnt/data/plan.csv

Modeling assumptions (explicit):
  - Each orders row is one demand line: (order_id, sku_id, branch_id, demand_qty, order_date, due_date).
  - Shipments can be split across days and supply locations.
  - A shipment decision is: (order_id, sku_id, from_location, to_location, ship_date, quantity, mode).
  - expected_arrival_date = ship_date + lead_time_days (from transport_costs).
  - Inventory is consumed at from_location_id at ship time.
  - Capacity constraints apply per from_location_id per ship_date:
      * max_units: total shipped units that day
      * max_shipments: number of distinct shipment decisions that day
    Weight is applied only if inventory has a per-unit weight column and capacity has max_weight_kg.
  - Transport cost per decision: fixed_cost + cost_per_unit * quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

import data_io


OUTPUT_COLS = [
    "decision_id",
    "order_id",
    "item_id",
    "from_location_id",
    "to_location_id",
    "ship_date",
    "quantity",
    "mode",
    "expected_arrival_date",
    "notes",
]


def _priority_rank(x: object) -> int:
    """Lower is higher priority."""
    s = str(x).strip().lower()
    return {
        "critical": 0,
        "urgent": 0,
        "high": 1,
        "normal": 2,
        "medium": 2,
        "low": 3,
    }.get(s, 9)


def _as_int_qty(x: object) -> int:
    """Best-effort conversion to a non-negative int."""
    try:
        v = float(x)
    except Exception:
        return 0
    if v <= 0:
        return 0
    # Deterministic rounding; demand_qty is expected to be integral in typical datasets.
    return int(pd.Series([v]).round().iloc[0])


@dataclass(frozen=True)
class ModeOption:
    mode: str
    lead_days: int
    cost_per_unit: float
    fixed_cost: float


def _build_lane_options(transport_costs: pd.DataFrame) -> Dict[Tuple[str, str], List[ModeOption]]:
    lanes: Dict[Tuple[str, str], List[ModeOption]] = {}
    for r in transport_costs.itertuples(index=False):
        key = (str(r.from_location_id), str(r.to_location_id))
        lanes.setdefault(key, []).append(
            ModeOption(
                mode=str(r.mode),
                lead_days=int(r.lead_time_days),
                cost_per_unit=float(r.cost_per_unit),
                fixed_cost=float(r.fixed_cost),
            )
        )
    # Deterministic ordering within a lane.
    for k in lanes:
        lanes[k].sort(key=lambda m: (m.cost_per_unit, m.fixed_cost, m.lead_days, m.mode))
    return lanes


def _best_mode(
    options: List[ModeOption], ship_date: pd.Timestamp, due_date: pd.Timestamp, on_time_required: bool
) -> Optional[ModeOption]:
    # Ranking required by prompt:
    #   a) on-time feasibility first
    #   b) then lower cost_per_unit, then lower fixed_cost
    if on_time_required:
        on_time = [m for m in options if ship_date + pd.Timedelta(days=m.lead_days) <= due_date]
        if not on_time:
            return None
        return min(on_time, key=lambda m: (m.cost_per_unit, m.fixed_cost, m.lead_days, m.mode))
    return min(options, key=lambda m: (m.cost_per_unit, m.fixed_cost, m.lead_days, m.mode))


def _lane_score(options: List[ModeOption]) -> Tuple[float, float, int, str]:
    """Score used to sort supply locations (cheaper lanes first)."""
    best = min(options, key=lambda m: (m.cost_per_unit, m.fixed_cost, m.lead_days, m.mode))
    return (best.cost_per_unit, best.fixed_cost, best.lead_days, best.mode)


def _init_inventory_available(inv: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    # Safety stock rule (explicit): if safety_stock exists, available = max(0, on_hand - safety_stock).
    has_ss = "safety_stock" in inv.columns
    avail: Dict[Tuple[str, str], float] = {}
    for r in inv.itertuples(index=False):
        loc = str(r.location_id)
        sku = str(r.sku_id)
        on_hand = float(r.on_hand_qty)
        ss = float(getattr(r, "safety_stock", 0.0)) if has_ss else 0.0
        avail[(loc, sku)] = max(0.0, on_hand - ss)
    return avail


def _init_capacity_state(
    cap: pd.DataFrame, use_weight: bool
) -> Tuple[Dict[Tuple[str, pd.Timestamp], Dict[str, float]], Dict[str, List[pd.Timestamp]]]:
    state: Dict[Tuple[str, pd.Timestamp], Dict[str, float]] = {}
    dates_by_loc: Dict[str, List[pd.Timestamp]] = {}
    for r in cap.itertuples(index=False):
        loc = str(r.location_id)
        d = pd.Timestamp(r.date).normalize()
        dates_by_loc.setdefault(loc, []).append(d)
        state[(loc, d)] = {
            "units": float(r.max_units),
            "shipments": float(r.max_shipments),
        }
        if use_weight and "max_weight_kg" in cap.columns:
            state[(loc, d)]["weight"] = float(getattr(r, "max_weight_kg", 0.0))
    for loc in dates_by_loc:
        dates_by_loc[loc] = sorted(set(dates_by_loc[loc]))
    return state, dates_by_loc


def planner(
    orders: pd.DataFrame,
    inventory: pd.DataFrame,
    capacity: pd.DataFrame,
    transport_costs: pd.DataFrame,
    weight_col: str | None,
) -> pd.DataFrame:
    """Return decisions DataFrame with OUTPUT_COLS."""

    lanes = _build_lane_options(transport_costs)
    inv_avail = _init_inventory_available(inventory)
    sku_to_locs: Dict[str, List[str]] = {}
    for (loc, sku), q in inv_avail.items():
        if q > 0:
            sku_to_locs.setdefault(sku, []).append(loc)
    for sku in sku_to_locs:
        sku_to_locs[sku] = sorted(set(sku_to_locs[sku]))

    use_weight = bool(weight_col) and ("max_weight_kg" in capacity.columns)
    weight_by_sku = (
        inventory.set_index("sku_id")[weight_col].to_dict() if use_weight and weight_col else {}
    )
    cap_state, cap_dates = _init_capacity_state(capacity, use_weight=use_weight)

    # For max_shipments, we track decisions (distinct shipment decisions) per (loc, date).
    # If we add quantity to an existing decision key, it does not consume another shipment slot.
    decision_index: Dict[Tuple[str, str, str, str, pd.Timestamp, str], int] = {}
    decisions: List[dict] = []
    order_to_decision_idxs: Dict[str, List[int]] = {}

    # Deterministic order processing.
    ords = orders.copy()
    if "priority" in ords.columns:
        ords["_priority_rank"] = ords["priority"].map(_priority_rank)
    else:
        ords["_priority_rank"] = 9
    ords = ords.sort_values(
        by=["due_date", "_priority_rank", "order_id"], kind="mergesort"
    ).reset_index(drop=True)

    for r in ords.itertuples(index=False):
        order_id = str(r.order_id)
        sku = str(r.sku_id)
        branch = str(r.branch_id)
        order_date = pd.Timestamp(r.order_date).normalize()
        due_date = pd.Timestamp(r.due_date).normalize()

        remaining = _as_int_qty(r.demand_qty)
        if remaining <= 0:
            continue

        # Candidate supply locations with available inventory.
        cand_locs = [loc for loc in sku_to_locs.get(sku, []) if inv_avail.get((loc, sku), 0) > 0]
        if not cand_locs:
            # No shipments possible; cannot emit a plan row with qty=0.
            continue

        # Filter to those with at least one lane to branch.
        loc_info = []
        for loc in cand_locs:
            opts = lanes.get((loc, branch))
            if not opts:
                continue
            # Compute if on-time is possible from this loc given capacity dates.
            dates = [d for d in cap_dates.get(loc, []) if d >= order_date]
            can_on_time = False
            for d in reversed(dates):
                if _best_mode(opts, d, due_date, on_time_required=True) is not None:
                    can_on_time = True
                    break
            loc_info.append((loc, can_on_time, _lane_score(opts)))

        if not loc_info:
            continue

        any_on_time_loc = any(can for _, can, _ in loc_info)
        # Sort: cheaper lanes first within each group.
        loc_info.sort(key=lambda t: (t[2], t[0]))
        on_time_locs = [(loc, can, sc) for (loc, can, sc) in loc_info if can]
        late_locs = [(loc, can, sc) for (loc, can, sc) in loc_info if not can]

        # Two-phase allocation:
        #   1) Try on-time capable supply first.
        #   2) If remaining > 0, allow late supply to reduce unfilled demand.
        loc_passes: List[List[Tuple[str, bool, Tuple[float, float, int, str]]]] = [on_time_locs]
        if any_on_time_loc:
            loc_passes.append(late_locs)
        else:
            loc_passes = [late_locs]

        for pass_locs in loc_passes:
            for loc, can_on_time, _score in pass_locs:
                if remaining <= 0:
                    break

                opts = lanes[(loc, branch)]
                dates = [d for d in cap_dates.get(loc, []) if d >= order_date]
                if not dates:
                    continue

                # Prefer shipping as late as possible while still arriving on time.
                on_time_dates = []
                for d in reversed(dates):
                    if _best_mode(opts, d, due_date, on_time_required=True) is not None:
                        on_time_dates.append(d)

                if on_time_dates:
                    date_iter = on_time_dates  # already descending
                    on_time_required = True
                else:
                    # No on-time option from this location: ship as early as possible to reduce lateness.
                    date_iter = sorted(dates)
                    on_time_required = False

                for ship_date in date_iter:
                    if remaining <= 0:
                        break
                    cap_key = (loc, ship_date)
                    if cap_key not in cap_state:
                        continue
                    cap_units = int(cap_state[cap_key]["units"])
                    if cap_units <= 0:
                        continue

                    # Pick mode.
                    mode_opt = _best_mode(opts, ship_date, due_date, on_time_required=on_time_required)
                    if mode_opt is None:
                        continue

                    # Available inventory and capacity.
                    inv_key = (loc, sku)
                    inv_here = int(inv_avail.get(inv_key, 0.0))
                    if inv_here <= 0:
                        continue

                    qty = min(remaining, inv_here, cap_units)
                    if qty <= 0:
                        continue

                    # Weight constraint (optional).
                    if use_weight:
                        w = float(weight_by_sku.get(sku, 0.0))
                        max_w = float(cap_state[cap_key].get("weight", 0.0))
                        if w > 0 and max_w > 0:
                            qty = min(qty, int(max_w // w))
                            if qty <= 0:
                                continue

                    # max_shipments constraint: only if this is a new decision key.
                    decision_key = (order_id, sku, loc, branch, ship_date, mode_opt.mode)
                    if decision_key not in decision_index:
                        if int(cap_state[cap_key]["shipments"]) <= 0:
                            continue
                        cap_state[cap_key]["shipments"] -= 1

                    # Consume capacity + inventory.
                    cap_state[cap_key]["units"] -= qty
                    inv_avail[inv_key] -= qty
                    if use_weight:
                        w = float(weight_by_sku.get(sku, 0.0))
                        if w > 0 and "weight" in cap_state[cap_key]:
                            cap_state[cap_key]["weight"] -= qty * w

                    remaining -= qty

                    arrival = (ship_date + pd.Timedelta(days=mode_opt.lead_days)).normalize()
                    is_on_time = arrival <= due_date
                    note_parts = ["on_time" if is_on_time else "late"]
                    if not on_time_required:
                        note_parts.append("no_on_time_option")

                    if decision_key in decision_index:
                        idx = decision_index[decision_key]
                        decisions[idx]["quantity"] += int(qty)
                    else:
                        idx = len(decisions)
                        decision_index[decision_key] = idx
                        decisions.append(
                            {
                                "decision_id": "",  # filled later
                                "order_id": order_id,
                                "item_id": sku,
                                "from_location_id": loc,
                                "to_location_id": branch,
                                "ship_date": ship_date.date().isoformat(),
                                "quantity": int(qty),
                                "mode": mode_opt.mode,
                                "expected_arrival_date": arrival.date().isoformat(),
                                "notes": ";".join(note_parts),
                            }
                        )
                        order_to_decision_idxs.setdefault(order_id, []).append(idx)

            if remaining <= 0:
                break

        # If demand remains, mark partial on the last decision row (if any).
        if remaining > 0:
            idxs = order_to_decision_idxs.get(order_id, [])
            if idxs:
                last_idx = idxs[-1]
                suffix = f"partial_unfilled={remaining}"
                if decisions[last_idx]["notes"]:
                    decisions[last_idx]["notes"] += ";" + suffix
                else:
                    decisions[last_idx]["notes"] = suffix

    out = pd.DataFrame(decisions, columns=OUTPUT_COLS)
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    # decision_id sequential
    out = out.reset_index(drop=True)
    out["decision_id"] = [f"D{(i+1):06d}" for i in range(len(out))]
    out["quantity"] = out["quantity"].astype(int)

    # Never write zero or negative quantities (guard against any bug).
    out = out[out["quantity"] > 0].copy()
    return out[OUTPUT_COLS]


def _write_empty_plan(path: str) -> None:
    pd.DataFrame(columns=OUTPUT_COLS).to_csv(path, index=False)


def main() -> None:
    orders_path = "data/orders.csv"
    inv_path = "data/inventory.csv"
    cap_path = "data/capacity.csv"
    tc_path = "data/transport_costs.csv"
    out_path = "data/plan.csv"

    try:
        orders = data_io.load_orders(orders_path)
        inventory = data_io.load_inventory(inv_path)
        capacity = data_io.load_capacity(cap_path)
        transport_costs = data_io.load_transport_costs(tc_path)

        validated = data_io.validate_all(orders, inventory, capacity, transport_costs)
        
        for w in validated.warnings:
            print("WARNING:", w)

        # print(validated.orders.head(10))
        exit(0)
        plan = planner(
            validated.orders,
            validated.inventory,
            validated.capacity,
            validated.transport_costs,
            validated.weight_col,
        )
        plan.to_csv(out_path, index=False)
    except Exception as e:
        print("ERROR:", e)
        _write_empty_plan(out_path)


if __name__ == "__main__":
    main()
