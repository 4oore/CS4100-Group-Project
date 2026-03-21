import random
import pandas as pd


def _best_multiplier(row: pd.Series, slot_cal_target: float, portion_min: float, portion_max: float) -> float:
    """
    Compute the serving multiplier that hits a per-slot calorie target.
    Clamped to [portion_min, portion_max].
    """
    kcal = row["energy_kcal"]
    if kcal <= 0:
        return portion_min
    mult = slot_cal_target / kcal
    return max(portion_min, min(portion_max, round(mult, 2)))


def _gap_score(row: pd.Series, mult: float, slot_targets: dict) -> float:
    """
    Normalized squared error between what this food provides (at mult)
    and the per-slot nutrient targets. Lower is better.
    """
    score = 0.0
    for n, target in slot_targets.items():
        if target > 0:
            score += ((row[n] * mult - target) / target) ** 2
    return score


def greedy_init(
    foods_df: pd.DataFrame,
    logged: dict,
    targets: dict,
    n_slots: int,
    penalty_cfg: dict,
    candidate_pool_size: int = 50,
    seed: int = 42,
) -> list[tuple[int, float]]:
    """
    Build an initial meal plan using a greedy slot-by-slot strategy.

    For each slot:
      1. Spread remaining nutrient targets evenly across unfilled slots.
      2. Sample `candidate_pool_size` foods at random.
      3. For each candidate, compute the multiplier that best hits the
         per-slot calorie target, then score it against all per-slot targets.
      4. Apply a soft penalty for repeating a food category already in the plan.
      5. Select the lowest-scoring candidate and append it to the state.
      6. Subtract its contribution from the remaining targets before the next slot.

    Parameters
    ----------
    foods_df : pd.DataFrame
        Cleaned food database (output of preprocess.py).
    logged : dict
        Nutrients already consumed today, e.g. {"energy_kcal": 400, ...}.
    targets : dict
        Full-day nutrient targets, e.g. {"energy_kcal": 2000, "protein_g": 150, ...}.
    n_slots : int
        Number of remaining meal slots to fill (e.g. 3 for breakfast/lunch/dinner).
    penalty_cfg : dict
        Must contain "portion_min", "portion_max", and optionally "variety_penalty".
        "variety_penalty" is a score multiplier applied when a candidate's category
        has already appeared in the plan (default 1.2 = 20% score penalty).
    candidate_pool_size : int
        How many randomly sampled foods to evaluate per slot. Higher values
        improve initial plan quality at the cost of initialization time.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of (food_id, multiplier) tuples, one per slot.
    """
    rng = random.Random(seed)
    portion_min = penalty_cfg["portion_min"]
    portion_max = penalty_cfg["portion_max"]
    variety_penalty = penalty_cfg.get("variety_penalty", 1.2)

    n_foods = len(foods_df)

    # Remaining nutrient budget after what's already been logged today
    rem_targets = {k: max(0.0, targets[k] - logged.get(k, 0.0)) for k in targets}

    state: list[tuple[int, float]] = []
    used_food_ids: set[int] = set()
    used_categories: set[str] = set()

    for slot_idx in range(n_slots):
        slots_left = n_slots - slot_idx

        # Spread the remaining budget evenly across unfilled slots
        slot_targets = {k: v / slots_left for k, v in rem_targets.items()}
        slot_cal_target = slot_targets.get("energy_kcal", 500.0)

        # Sample a pool of candidates, excluding foods already in the plan
        pool = [i for i in rng.sample(range(n_foods), min(candidate_pool_size + len(used_food_ids), n_foods))
                if i not in used_food_ids][:candidate_pool_size]

        # Fallback: if the pool is exhausted (very small dataset), allow repeats
        if not pool:
            pool = rng.sample(range(n_foods), min(candidate_pool_size, n_foods))

        best_food_id = None
        best_mult = portion_min
        best_score = float("inf")

        for food_id in pool:
            row = foods_df.iloc[food_id]
            mult = _best_multiplier(row, slot_cal_target, portion_min, portion_max)
            score = _gap_score(row, mult, slot_targets)

            # Soft variety enforcement: penalize foods from already-used categories
            if row["category"] in used_categories:
                score *= variety_penalty

            if score < best_score:
                best_score = score
                best_food_id = food_id
                best_mult = mult

        # Should never happen, but guard against an empty pool
        if best_food_id is None:
            best_food_id = rng.randrange(n_foods)
            row = foods_df.iloc[best_food_id]
            best_mult = _best_multiplier(row, slot_cal_target, portion_min, portion_max)

        state.append((best_food_id, best_mult))

        # Deduct chosen food's contribution from the remaining budget
        chosen_row = foods_df.iloc[best_food_id]
        for k in rem_targets:
            rem_targets[k] = max(0.0, rem_targets[k] - chosen_row[k] * best_mult)

        used_food_ids.add(best_food_id)
        used_categories.add(chosen_row["category"])

    return state