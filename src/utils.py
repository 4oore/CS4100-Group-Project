import pandas as pd


def print_plan(
    best_state: list,
    remaining_slots: list,
    foods_df: pd.DataFrame,
    targets: dict,
    upper_limits: dict,
    logged_today: dict,
) -> None:
    """Pretty-print the optimised meal plan and nutrient totals."""

    print(f"\nLogged today: {logged_today}")
    print(f"Slots to plan: {remaining_slots}\n")

    print("Meal Plan")
    nutrients = list(targets.keys()) + list(upper_limits.keys())
    total = {n: 0.0 for n in nutrients}

    for slot, (food_id, mult) in zip(remaining_slots, best_state):
        row = foods_df.iloc[food_id]
        grams = mult * row["basis_g"]
        print(f"\n  {slot.upper()} ({grams:.0f}g):")
        print(f"    {row['name']}  [{row['category']}]")
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
            status = f"  ({'over' if diff > 0 else 'under'} by {abs(diff):.1f})"
            print(f"  {n}: {combined:.1f} / {target:.1f}{status}")
        elif n in upper_limits:
            limit = upper_limits[n]
            status = "  EXCEEDED" if combined > limit else "  within limit"
            print(f"  {n}: {combined:.1f} / {limit:.1f}{status}")
