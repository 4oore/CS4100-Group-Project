import pandas as pd
from .score import fitness
from deep_translator import GoogleTranslator  # type: ignore

translator = GoogleTranslator(source="auto", target="en")
_translation_cache: dict[str, str] = {}


def translate_for_display(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text

    if text in _translation_cache:
        return _translation_cache[text]

    try:
        translated = translator.translate(text)
        _translation_cache[text] = translated
        return translated
    except Exception:
        return text


def print_plan(
    best_state: list,
    remaining_slots: list[str],
    foods_df: pd.DataFrame,
    targets: dict[str, float],
    upper_limits: dict[str, float],
    logged_today: dict[str, float],
    score_cfg: dict[str, float],
    preferences: dict[str, set[str]] | None = None,
    logged_categories: list[str] | None = None,
    follow_history: dict[str, dict[str, int]] | None = None,
) -> None:
    """Pretty-print the optimized meal plan and nutrient totals."""

    print(f"\nLogged today: {logged_today}")
    print(f"Slots to plan: {remaining_slots}\n")

    print("Recommended Meal Plan")
    nutrients = list(targets.keys()) + list(upper_limits.keys())
    total = {n: 0.0 for n in nutrients}

    for slot, (food_id, mult) in zip(remaining_slots, best_state):
        row = foods_df.iloc[food_id]
        grams = mult * row["basis_g"]

        raw_name = str(row["name"]).strip()
        raw_category = str(row["category"]).split(",")[0].strip()

        name = translate_for_display(raw_name)
        category = translate_for_display(raw_category)

        print(f"\n  {slot.title()} ({grams:.0f} g)")
        print(f"    {name}")
        print(f"    Category: {category}")

        for n in nutrients:
            val = row[n] * mult
            total[n] += val
            print(f"      {n}: {val:.1f}")

    print("\nFull day totals (logged + planned)")
    for n in nutrients:
        logged_val = logged_today.get(n, 0.0)
        combined = logged_val + total[n]

        if n in targets:
            target = targets[n]
            diff = combined - target
            status = f"({'over' if diff > 0 else 'under'} by {abs(diff):.1f})"
            print(f"  {n}: {combined:.1f} / {target:.1f}  {status}")
        elif n in upper_limits:
            limit = upper_limits[n]
            status = "EXCEEDED" if combined > limit else "within limit"
            print(f"  {n}: {combined:.1f} / {limit:.1f}  {status}")

    fit = fitness(
        state=best_state,
        remaining_slots=remaining_slots,
        foods_df=foods_df,
        logged=logged_today,
        targets=targets,
        upper_limits=upper_limits,
        score_cfg=score_cfg,
        preferences=preferences,
        logged_categories=logged_categories,
        follow_history=follow_history,
    )

    print("\nFitness Breakdown")
    print(f"  NutritionScore:         {fit['nutrition']:.3f}")
    print(f"  VarietyScore:           {fit['variety']:.3f}")
    print(f"  MealCompatibilityScore: {fit['compatibility']:.3f}")
    print(f"  UserPreferenceScore:    {fit['preference']:.3f}")
    print(f"  FollowThroughScore:     {fit['follow']:.3f}")
    print(f"  Final Fitness:          {fit['fitness']:.3f}")
    print(f"  Energy (1 - fitness):   {1.0 - fit['fitness']:.3f}")