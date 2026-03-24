from __future__ import annotations
from collections import Counter
from typing import Optional
import pandas as pd

State = list[tuple[int, float]]  # List of (food_id, multiplier) pairs

# ********* GENETIC ALGORITHM SCORING FUNCTIONS *********

def _safe_normalized_closeness(actual: float, target: float) -> float:
    """
    Returns a score in [0, 1].
    1 means exact match, lower means farther away.
    """
    if target <= 0:
        return 1.0
    return max(0.0, 1.0 - abs(actual - target) / target)


def _safe_upper_limit_score(actual: float, limit: float, overshoot_weight: float = 2.0) -> float:
    """
    Score in [0, 1] for nutrients with upper bounds (e.g. sodium, sugar).
    If under the limit -> 1.0
    If over the limit -> decreases as overshoot grows.
    """
    if limit <= 0:
        return 1.0
    if actual <= limit:
        return 1.0
    overshoot_ratio = (actual - limit) / limit
    return max(0.0, 1.0 - overshoot_weight * overshoot_ratio)


def _category_to_slot_score(category: str, slot: str) -> float:
    """
    Rule-based food-meal compatibility score in [0, 1].
    Strongly penalizes snack/breakfast foods in lunch/dinner slots.
    """
    c = category.lower()
    s = slot.lower()

    breakfast_keywords = [
        "breakfast",
        "breakfast cereals",
        "cereal",
        "cereals",
        "oat",
        "oats",
        "oatmeal",
        "yogurt",
        "granola",
        "muesli",
        "porridge",
        "pancake",
        "waffle",
    ]

    snack_keywords = [
        "snack",
        "snacks",
        "cookies",
        "biscuits",
        "cracker",
        "crackers",
        "chips",
        "crisps",
        "rice cakes",
        "puffed",
        "bars",
        "appetizers",
        "salty snacks",
        "nuts",
    ]

    side_keywords = [
        "fries",
        "onion rings",
        "chips",
        "crisps",
        "appetizers",
        "salty snacks",
        "crackers",
        "biscuits",
    ]

    meal_keywords = [
        "meal",
        "meals",
        "prepared meals",
        "chicken",
        "beef",
        "pork",
        "fish",
        "rice",
        "pasta",
        "pizza",
        "salad",
        "wrap",
        "sandwich",
        "frozen meals",
        "poultry",
        "dinner",
        "lunch",
        "teriyaki",
    ]

    vegetable_keywords = [
        "vegetable",
        "vegetables",
        "soups",
        "prepared vegetables",
        "legumes",
        "beans",
    ]

    if s == "breakfast":
        if any(k in c for k in breakfast_keywords):
            return 1.0
        if any(k in c for k in snack_keywords):
            return 0.65
        if any(k in c for k in meal_keywords):
            return 0.20
        if any(k in c for k in vegetable_keywords):
            return 0.35
        return 0.35

    if s == "snack":
        if any(k in c for k in snack_keywords):
            return 1.0
        if any(k in c for k in breakfast_keywords):
            return 0.75
        if any(k in c for k in meal_keywords):
            return 0.20
        if any(k in c for k in vegetable_keywords):
            return 0.40
        return 0.40

    if s == "lunch":
        if any(k in c for k in meal_keywords):
            return 1.0
        if any(k in c for k in vegetable_keywords):
            return 0.75
        if any(k in c for k in side_keywords):
            return 0.05
        if any(k in c for k in breakfast_keywords):
            return 0.05
        if any(k in c for k in snack_keywords):
            return 0.02
        return 0.15

    if s == "dinner":
        if any(k in c for k in meal_keywords):
            return 1.0
        if any(k in c for k in vegetable_keywords):
            return 0.75
        if any(k in c for k in side_keywords):
            return 0.03
        if any(k in c for k in breakfast_keywords):
            return 0.02
        if any(k in c for k in snack_keywords):
            return 0.01
        return 0.10

    return 0.25


def _category_preference_score(category: str, preferences: Optional[dict[str, set[str]]]) -> float:
    """
    Returns a score in [0, 1] based on liked/disliked category keywords.
    If no preferences are provided, return neutral score = 0.5.
    """
    if preferences is None:
        return 0.5

    category_l = category.lower()
    liked = preferences.get("liked_categories", set())
    disliked = preferences.get("disliked_categories", set())

    if any(k.lower() in category_l for k in disliked):
        return 0.0
    if any(k.lower() in category_l for k in liked):
        return 1.0
    return 0.5


def _planned_totals(
    state: State,
    foods_df: pd.DataFrame,
    nutrients: list[str],
) -> tuple[dict[str, float], list[str]]:
    planned = {n: 0.0 for n in nutrients}
    categories: list[str] = []

    for food_id, mult in state:
        row = foods_df.iloc[food_id]
        for n in nutrients:
            planned[n] += float(row[n]) * mult
        categories.append(str(row["category"]))

    return planned, categories


def nutrition_score(
    state: State,
    foods_df: pd.DataFrame,
    logged: dict[str, float],
    targets: dict[str, float],
    upper_limits: dict[str, float],
) -> float:
    """
    Proposal-aligned nutrition score in [0, 1].
    Uses logged intake + planned meals.
    """
    nutrients = list(targets.keys()) + list(upper_limits.keys())
    planned, _ = _planned_totals(state, foods_df, nutrients)

    scores: list[float] = []

    # target nutrients: energy, protein, carbs, fat, fiber
    for n, target in targets.items():
        actual = logged.get(n, 0.0) + planned[n]
        scores.append(_safe_normalized_closeness(actual, target))

    # upper-limit nutrients: sodium, sugar
    for n, limit in upper_limits.items():
        actual = logged.get(n, 0.0) + planned[n]
        scores.append(_safe_upper_limit_score(actual, limit, overshoot_weight=2.5))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def variety_score(
    state: State,
    foods_df: pd.DataFrame,
    logged_categories: Optional[list[str]] = None,
) -> float:
    """
    Penalizes repeated foods/categories.
    Score in [0, 1], where 1 means high variety.
    """
    categories: list[str] = []
    food_ids: list[int] = []

    for food_id, _ in state:
        row = foods_df.iloc[food_id]
        categories.append(str(row["category"]))
        food_ids.append(food_id)

    if logged_categories:
        categories.extend(logged_categories)

    cat_counts = Counter(categories)
    food_counts = Counter(food_ids)

    repeated_categories_penalty = sum(max(0, c - 1) for c in cat_counts.values())
    repeated_foods_penalty = sum(max(0, c - 1) for c in food_counts.values())

    total_penalty = repeated_categories_penalty + repeated_foods_penalty

    return 1.0 / (1.0 + total_penalty)


def meal_compatibility_score(
    state: State,
    remaining_slots: list[str],
    foods_df: pd.DataFrame,
) -> float:
    """
    Score in [0, 1] measuring whether foods fit their meal slots.
    """
    if not state:
        return 0.0

    scores: list[float] = []
    for slot, (food_id, _) in zip(remaining_slots, state):
        row = foods_df.iloc[food_id]
        category = str(row["category"])
        scores.append(_category_to_slot_score(category, slot))

    return sum(scores) / len(scores)


def user_preference_score(
    state: State,
    foods_df: pd.DataFrame,
    preferences: Optional[dict[str, set[str]]] = None,
) -> float:
    """
    Score in [0, 1] based on simple liked/disliked category keywords.
    """
    if not state:
        return 0.0

    scores: list[float] = []
    for food_id, _ in state:
        row = foods_df.iloc[food_id]
        category = str(row["category"])
        scores.append(_category_preference_score(category, preferences))

    return sum(scores) / len(scores)

def follow_through_score(
    state: State,
    foods_df: pd.DataFrame,
    follow_history: Optional[dict[str, dict[str, int]]] = None,
) -> float:
    """
    Score in [0, 1] based on whether recommended categories were historically followed.
    follow_history format:
    {
        "accepted": {"fruit": 3, "yogurt": 2},
        "rejected": {"dessert": 4, "pork": 1}
    }

    If no history exists for a category, return neutral score = 0.5.
    """
    if not state:
        return 0.0

    if follow_history is None:
        return 0.5

    accepted = follow_history.get("accepted", {})
    rejected = follow_history.get("rejected", {})

    scores: list[float] = []

    for food_id, _ in state:
        row = foods_df.iloc[food_id]
        category = str(row["category"]).lower()

        a = accepted.get(category, 0)
        r = rejected.get(category, 0)

        total = a + r
        if total == 0:
            scores.append(0.5)  # neutral if unseen
        else:
            scores.append(a / total)

    return sum(scores) / len(scores)


def fitness(
    state: State,
    remaining_slots: list[str],
    foods_df: pd.DataFrame,
    logged: dict[str, float],
    targets: dict[str, float],
    upper_limits: dict[str, float],
    score_cfg: dict[str, float],
    preferences: Optional[dict[str, set[str]]] = None,
    logged_categories: Optional[list[str]] = None,
    follow_history: Optional[dict[str, dict[str, int]]] = None,
) -> dict[str, float]:
    """
    Returns all sub-scores plus total fitness.
    """
    nutrition = nutrition_score(state, foods_df, logged, targets, upper_limits)
    variety = variety_score(state, foods_df, logged_categories=logged_categories)
    compatibility = meal_compatibility_score(state, remaining_slots, foods_df)
    follow = follow_through_score(state, foods_df, follow_history=follow_history)
    preference = user_preference_score(state, foods_df, preferences=preferences)

    total = (
        score_cfg["w_nutrition"] * nutrition
        + score_cfg["w_variety"] * variety
        + score_cfg["w_compatibility"] * compatibility
        + score_cfg["w_preference"] * preference
        + score_cfg["w_follow"] * follow
    )

    return {
        "nutrition": nutrition,
        "variety": variety,
        "compatibility": compatibility,
        "preference": preference,
        "follow": follow,
        "fitness": total,
    }


def energy(
    state: State,
    remaining_slots: list[str],
    foods_df: pd.DataFrame,
    logged: dict[str, float],
    targets: dict[str, float],
    upper_limits: dict[str, float],
    score_cfg: dict[str, float],
    preferences: Optional[dict[str, set[str]]] = None,
    logged_categories: Optional[list[str]] = None,
    follow_history: Optional[dict[str, dict[str, int]]] = None,
) -> float:
    """
    Lower is better for SA/GA.
    We minimize energy = 1 - fitness.
    """
    fit = fitness(
        state=state,
        remaining_slots=remaining_slots,
        foods_df=foods_df,
        logged=logged,
        targets=targets,
        upper_limits=upper_limits,
        score_cfg=score_cfg,
        preferences=preferences,
        logged_categories=logged_categories,
        follow_history=follow_history,
    )["fitness"]
    return 1.0 - fit