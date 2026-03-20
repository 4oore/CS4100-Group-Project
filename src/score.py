from collections import Counter
import pandas as pd


def energy(
    state: list,
    foods_df: pd.DataFrame,
    logged: dict,
    targets: dict,
    upper_limits: dict,
    penalty_cfg: dict[str, float],
) -> float:
    rem_targets = {k: max(0.0, targets[k] - logged.get(k, 0.0)) for k in targets}
    rem_limits = {
        k: max(0.0, upper_limits[k] - logged.get(k, 0.0)) for k in upper_limits
    }
    nutrients = list(targets.keys()) + list(upper_limits.keys())

    # Sum planned nutrients across all slots
    planned = {n: 0.0 for n in nutrients}
    categories = []
    slot_cals = []
    for food_id, mult in state:
        row = foods_df.iloc[food_id]
        for n in nutrients:
            planned[n] += row[n] * mult
        categories.append(row["category"])
        slot_cals.append(row["energy_kcal"] * mult)

    # squared error from remaining targets, normalized
    gap = 0.0
    for n, target in rem_targets.items():
        if target > 0:
            gap += ((planned[n] - target) / target) ** 2

    # only penalize if plan exceeds remaining allowance
    exceed = 0.0
    for n, limit in rem_limits.items():
        if planned[n] > limit:
            exceed += ((planned[n] - limit) / max(limit, 1.0)) ** 2

    # penalize repeated categories in the same plan
    cat_counts = Counter(categories)
    variety = sum(max(0, c - 1) ** 2 for c in cat_counts.values())

    # penalize multipliers outside sensible range
    portion = 0.0
    for _, mult in state:
        if mult < penalty_cfg["portion_min"]:
            portion += (penalty_cfg["portion_min"] - mult) ** 2
        elif mult > penalty_cfg["portion_max"]:
            portion += (mult - penalty_cfg["portion_max"]) ** 2

    # Penalize uneven calorie distribution across slots
    balance = 0.0
    if len(slot_cals) > 1:
        mean_cal = sum(slot_cals) / len(slot_cals)
        if mean_cal > 0:
            balance = sum((c - mean_cal) ** 2 for c in slot_cals) / (mean_cal**2)

    return (
        penalty_cfg["w_gap"] * gap
        + penalty_cfg["w_exceed"] * exceed
        + penalty_cfg["w_variety"] * variety
        + penalty_cfg["w_portion"] * portion
        + penalty_cfg["w_balance"] * balance
    )
