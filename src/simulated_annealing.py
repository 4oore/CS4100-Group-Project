import copy
import math
import random
import pandas as pd
from .score import energy


def neighbour(state: list, foods_df: pd.DataFrame, rng: random.Random) -> list:
    s = copy.deepcopy(state)
    move = rng.choice(["replace", "swap", "resize"])

    if move == "replace":
        idx = rng.randrange(len(s))
        new_food = rng.randrange(len(foods_df))
        s[idx] = (new_food, s[idx][1])

    elif move == "swap":
        in_plan = {food_id for food_id, _ in s}
        outside = [i for i in range(len(foods_df)) if i not in in_plan]
        if outside:
            idx = rng.randrange(len(s))
            new_food = rng.choice(outside)
            s[idx] = (new_food, s[idx][1])

    elif move == "resize":
        idx = rng.randrange(len(s))
        delta = rng.uniform(-0.5, 0.5)
        new_mult = max(0.1, s[idx][1] + delta)
        s[idx] = (s[idx][0], round(new_mult, 2))

    return s


def simulated_annealing(
    foods_df: pd.DataFrame,
    logged: dict,
    initial_state: list,
    targets: dict,
    upper_limits: dict,
    remaining_slots: list[str],
    sa_cfg: dict,
    score_cfg: dict,
    preferences: dict[str, set[str]] | None = None,
    logged_categories: list[str] | None = None,
    follow_history: dict[str, dict[str, int]] | None = None,
) -> list:
    rng = random.Random(sa_cfg["seed"])

    state = copy.deepcopy(initial_state)
    best = copy.deepcopy(state)
    best_e = energy(
        state,
        remaining_slots,
        foods_df,
        logged,
        targets,
        upper_limits,
        score_cfg,
        preferences=preferences,
        logged_categories=logged_categories,
        follow_history=follow_history,
    )

    T = sa_cfg["T_start"]
    curr_e = best_e
    for step in range(sa_cfg["max_steps"]):
        candidate = neighbour(state, foods_df, rng)
        cand_e = energy(
            candidate,
            remaining_slots,
            foods_df,
            logged,
            targets,
            upper_limits,
            score_cfg,
            preferences=preferences,
            logged_categories=logged_categories,
            follow_history=follow_history,
        )
        delta = cand_e - curr_e

        if delta < 0 or rng.random() < math.exp(-delta / T):
            state = candidate
            curr_e = cand_e
            if cand_e < best_e:
                best = copy.deepcopy(state)
                best_e = cand_e

        T = max(sa_cfg["T_end"], T * sa_cfg["alpha"])

        if step % 2000 == 0:
            print(f"step {step}, T={T:.2f}, best_energy={best_e:.4f}")

    return best
