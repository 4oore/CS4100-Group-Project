import copy
import math
import random
import pandas as pd
from .score import energy


def neighbour(state: list, foods_df: pd.DataFrame, rng: random.Random) -> list:
    """
    Generate a neighbouring state by applying one of three random moves:
      - replace: swap a slot's food for any random food from the dataset
      - swap:    exchange two foods between their slots within the plan
      - resize:  nudge a slot's portion multiplier by a random amount in [-0.5, 0.5]
    """
    s = copy.deepcopy(state)
    move = rng.choice(["replace", "swap", "resize"])

    if move == "replace":
        idx = rng.randrange(len(s))
        new_food = rng.randrange(len(foods_df))
        s[idx] = (new_food, s[idx][1])

    elif move == "swap" and len(s) >= 2:
        i, j = rng.sample(range(len(s)), 2)
        s[i], s[j] = s[j], s[i]

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
    """
    Run simulated annealing to find the optimal meal plan.

    At each step a neighbour state is generated and accepted if it improves the
    energy or with probability exp(-delta/T) if it does not, allowing the search
    to escape local minima. Temperature is decayed by factor alpha each step
    until T_end, gradually shifting from exploration to exploitation.

    Args:
        foods_df:          DataFrame of all available foods.
        logged:            Nutrients already consumed today.
        initial_state:     Starting list of (food_index, multiplier) pairs.
        targets:           Daily nutrient targets (e.g. energy_kcal, protein_g).
        upper_limits:      Daily nutrient upper limits (e.g. sodium_mg, sugar_g).
        remaining_slots:   Meal slots still to be planned (e.g. ["lunch", "dinner"]).
        sa_cfg:            SA hyperparameters: T_start, T_end, alpha, max_steps, seed.
        score_cfg:         Scoring weights and portion bounds.
        preferences:       Optional liked/disliked category sets.
        logged_categories: Categories already eaten today, for variety scoring.
        follow_history:    Accepted/rejected category counts from past sessions.

    Returns:
        Best state found as a list of (food_index, multiplier) pairs.
    """
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
